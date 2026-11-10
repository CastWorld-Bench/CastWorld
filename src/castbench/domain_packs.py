"""Content-verified domain pack loader; all executable adapters are built-in and allowlisted."""

from __future__ import annotations

from pathlib import Path

from .common import digest, load_json
from .contracts import PublicBundle
from .domains import domain_registry


def load_domain_episodes(directory, episode_ids):
    root = Path(directory).resolve()
    manifest = load_json(root / "manifest.json")
    if manifest.get("schema_version") != "castbench.domain_pack/1.0":
        raise ValueError("Unsupported domain pack schema")
    if manifest.get("status") != "ready":
        raise ValueError("Domain pack is not approved for evaluation")

    def resource(ref):
        path = (root / ref["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Pack resource must stay inside the public pack")
        data = load_json(path)
        if digest(data) != ref["sha256"]:
            raise ValueError(f"Domain pack resource hash mismatch: {ref['path']}")
        return data

    episode_ids = list(episode_ids)
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Episode request must be nonempty and unique")
    listed = {}
    for item in manifest["episodes"]:
        episode_id = item["episode_id"]
        if episode_id in listed:
            raise ValueError("Episode is not uniquely listed in this pack")
        listed[episode_id] = item
    if not set(episode_ids) <= set(listed):
        raise ValueError("Episode is not uniquely listed in this pack")
    domain = manifest["domain_id"]
    shared = resource(manifest["shared"])
    cases = [resource(item) for item in shared["training_cases"]]
    if domain.startswith("dataset_"):
        from .dataset_domains import dataset_registry

        registry = dataset_registry(domain, shared["profile"])
    else:
        registry = domain_registry(domain)
    current_tools = {k: v for k, v in registry.manifests.items() if k in registry.extensions}
    if digest(current_tools) != manifest["tool_manifest_hash"]:
        raise ValueError("Installed domain tool version differs from frozen pack")
    loaded = {}
    shared_models = None
    shared_compatibility = None
    for episode_id in episode_ids:
        # The top-level dict is copied because samples/knowledge are attached below.
        # Nested resource values stay read-only and Pydantic builds independent models.
        data = dict(resource(listed[episode_id]))
        if shared.get("forecast_evaluation") is not None:
            if data.get("forecast_evaluation") not in (None, shared["forecast_evaluation"]):
                raise ValueError("Episode and shared forecast metric profiles differ")
            data["forecast_evaluation"] = shared["forecast_evaluation"]
        if data["episode"]["episode_id"] != episode_id:
            raise ValueError("Episode identity differs from manifest")
        if data["episode"]["domain_id"] != domain:
            raise ValueError("Episode domain differs from manifest")
        if shared_models is None:
            data.update(samples=cases, knowledge=shared["knowledge"])
            bundle = PublicBundle.model_validate(data)
            shared_models = {"samples": bundle.samples, "knowledge": bundle.knowledge}
            shared_compatibility = digest(
                {
                    "frequency": bundle.episode.task.frequency,
                    "sample_library_cutoff": bundle.sample_library_cutoff,
                    "variables": [
                        variable.model_dump(exclude={"values"}) for variable in bundle.variables
                    ],
                }
            )
        else:
            # Validate all Episode-specific fields without rebuilding and rechecking
            # the same potentially large sample/knowledge library. Reuse is allowed
            # only under the exact schema, frequency and cutoff that validated it.
            data.update(samples=[], knowledge=[])
            bundle = PublicBundle.model_validate(data)
            compatibility = digest(
                {
                    "frequency": bundle.episode.task.frequency,
                    "sample_library_cutoff": bundle.sample_library_cutoff,
                    "variables": [
                        variable.model_dump(exclude={"values"}) for variable in bundle.variables
                    ],
                }
            )
            if compatibility == shared_compatibility:
                bundle = bundle.model_copy(
                    update={name: list(values) for name, values in shared_models.items()}
                )
            else:
                data.update(samples=cases, knowledge=shared["knowledge"])
                bundle = PublicBundle.model_validate(data)
        loaded[episode_id] = (bundle, registry)
    return loaded


def load_domain_episode(directory, episode_id):
    return load_domain_episodes(directory, [episode_id])[episode_id]
