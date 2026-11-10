from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .common import AgentError, InfrastructureError, digest, json_text


class Store:
    """Immutable blobs plus a transactional journal; hashes alone confer no read permission."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "journal.sqlite"
        with self.connection() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS runs(
              run_id TEXT PRIMARY KEY, config TEXT NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS blobs(hash TEXT PRIMARY KEY, content TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS grants(run_id TEXT, hash TEXT, PRIMARY KEY(run_id,hash));
            CREATE TABLE IF NOT EXISTS attempts(
              id TEXT PRIMARY KEY, run_id TEXT, turn INTEGER, request TEXT,
              response TEXT, complete INTEGER NOT NULL DEFAULT 0, error TEXT);
            CREATE TABLE IF NOT EXISTS accepted(
              run_id TEXT, turn INTEGER, attempt_id TEXT, PRIMARY KEY(run_id,turn));
            CREATE TABLE IF NOT EXISTS staged(
              run_id TEXT, turn INTEGER, call_key TEXT, value TEXT,
              PRIMARY KEY(run_id,turn,call_key));
            CREATE TABLE IF NOT EXISTS commits(
              run_id TEXT, turn INTEGER, state TEXT, response TEXT, event TEXT,
              PRIMARY KEY(run_id,turn));
            CREATE TABLE IF NOT EXISTS audit(
              seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, kind TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS model_cache(key TEXT PRIMARY KEY, value TEXT);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def writer(self, run_id):
        # OS locks release on process death; unlike TTL locks they cannot expire mid-inference.
        lock_path = self.root / (digest(run_id).split(":")[1] + ".lock")
        with lock_path.open("a+b") as lock:
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                import os

                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise InfrastructureError("This Episode already has an active writer") from exc
            try:
                yield
            finally:
                lock.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def start(self, run_id, config, initial):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT config FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row and row[0] != json_text(config):
                raise InfrastructureError("Frozen run configuration differs; use a new run_id")
            db.execute(
                "INSERT OR IGNORE INTO runs VALUES(?,?,?)", (run_id, json_text(config), json_text(initial))
            )

    def get(self, run_id):
        with self.connection() as db:
            row = db.execute("SELECT state FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            raise InfrastructureError("Unknown run_id")
        return json.loads(row[0])

    def migrate_capacity_config(self, run_id, expected_hash, config):
        """Explicit continuation only; preserve old requests, commits and configuration in audit."""
        config = json.loads(json_text(config))
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT config,state FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise InfrastructureError("Continuation Episode is missing")
            old, state = map(json.loads, row)
            if old == config:
                return
            if digest(old) != expected_hash or state["status"] != "running":
                raise InfrastructureError("Continuation configuration hash/status mismatch")
            allowed = {"catalog_hash", "context_policy", "capacity_recovery", "provider"}
            changed = {k for k in old.keys() | config.keys() if old.get(k) != config.get(k)}
            if changed - allowed:
                raise InfrastructureError(f"Not a capacity-only continuation: {sorted(changed - allowed)}")
            provider_allowed = {
                "context_window_tokens",
                "max_output_tokens",
                "capacity_recovery",
                "timeout_seconds",
            }
            if old["provider"].get("adapter") != config["provider"].get("adapter"):
                raise InfrastructureError("Provider adapter changed during continuation")
            before, after = old["provider"].get("profile", {}), config["provider"].get("profile", {})
            # Provider manifests contain the frozen profile, not secrets.
            provider_changes = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
            if provider_changes - provider_allowed:
                raise InfrastructureError(
                    f"Unapproved Provider continuation changes: {sorted(provider_changes)}"
                )
            for name in ("context_window_tokens", "max_output_tokens", "timeout_seconds"):
                if after.get(name, 0) < before.get(name, 0):
                    raise InfrastructureError("Capacity continuation cannot reduce an existing limit")
            if db.execute(
                "SELECT 1 FROM accepted WHERE run_id=? AND turn>?", (run_id, state["state_version"])
            ).fetchone():
                raise InfrastructureError(
                    "Commit the already accepted response before migrating configuration"
                )
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "after_turn": state["state_version"],
                "old_config": old,
                "new_config": config,
                "notice": "Mixed-capacity continuation, not a uniform-profile evaluation",
            }
            db.execute(
                "INSERT INTO audit(run_id,kind,payload) VALUES(?,?,?)",
                (run_id, "capacity_config_continuation", json_text(payload)),
            )
            db.execute("UPDATE runs SET config=? WHERE run_id=?", (json_text(config), run_id))

    def config(self, run_id):
        with self.connection() as db:
            row = db.execute("SELECT config FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(row[0])

    def audit(self, run_id, kind, payload):
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
        with self.connection() as db:
            db.execute(
                "INSERT INTO audit(run_id,kind,payload) VALUES(?,?,?)", (run_id, kind, json_text(payload))
            )

    def put_blob(self, content):
        ref = digest(content)
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO blobs VALUES(?,?)", (ref, json_text(content)))
        return ref

    def read_blob(self, run_id, ref):
        with self.connection() as db:
            row = db.execute(
                "SELECT content FROM blobs JOIN grants USING(hash) WHERE run_id=? AND hash=?", (run_id, ref)
            ).fetchone()
        if not row:
            raise AgentError("Artifact not authorized for this run", "PERMISSION_DENIED")
        content = json.loads(row[0])
        if digest(content) != ref:
            raise InfrastructureError("Artifact integrity failure")
        return content

    def prepare(self, run_id, turn, request):
        aid = uuid.uuid4().hex
        with self.connection() as db:
            db.execute(
                "INSERT INTO attempts(id,run_id,turn,request) VALUES(?,?,?,?)",
                (aid, run_id, turn, json_text(request)),
            )
        self.audit(run_id, "provider_prepared", {"attempt_id": aid, "turn": turn})
        return aid

    def receive(self, aid, raw, complete=True, error=None):
        # First complete response wins, even if it is malformed or a refusal.
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT run_id,turn,response FROM attempts WHERE id=?", (aid,)).fetchone()
            if row is None or row[2] is not None:
                raise InfrastructureError("Attempt cannot be overwritten")
            db.execute(
                "UPDATE attempts SET response=?,complete=?,error=? WHERE id=?",
                (json_text(raw), int(complete), error, aid),
            )
            if complete:
                db.execute("INSERT OR IGNORE INTO accepted VALUES(?,?,?)", (row[0], row[1], aid))
        self.audit(
            row[0],
            "provider_received",
            {"attempt_id": aid, "turn": row[1], "complete": complete, "error": error},
        )

    def accepted(self, run_id, turn):
        with self.connection() as db:
            row = db.execute(
                "SELECT response FROM attempts JOIN accepted ON id=attempt_id "
                "WHERE accepted.run_id=? AND accepted.turn=?",
                (run_id, turn),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def has_accepted(self, run_id, turn):
        with self.connection() as db:
            return (
                db.execute("SELECT 1 FROM accepted WHERE run_id=? AND turn=?", (run_id, turn)).fetchone()
                is not None
            )

    def stage(self, run_id, turn, key, value):
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO staged VALUES(?,?,?,?)", (run_id, turn, key, json_text(value)))

    def staged(self, run_id, turn, key):
        with self.connection() as db:
            row = db.execute(
                "SELECT value FROM staged WHERE run_id=? AND turn=? AND call_key=?", (run_id, turn, key)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def commit(self, run_id, expected, state, response, event, refs=()):
        turn = expected + 1
        event = {
            "event_id": f"{run_id}/{turn}",
            "state_before": expected,
            "state_after": turn,
            "episode_id": state["episode_id"],
            "actor": "agent",
            **event,
        }
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT response FROM commits WHERE run_id=? AND turn=?", (run_id, turn)
            ).fetchone()
            if existing:
                return json.loads(existing[0])
            current = json.loads(db.execute("SELECT state FROM runs WHERE run_id=?", (run_id,)).fetchone()[0])
            if current["state_version"] != expected or state["state_version"] != turn:
                raise InfrastructureError("State version conflict")
            for ref in refs:
                if db.execute("SELECT 1 FROM blobs WHERE hash=?", (ref,)).fetchone() is None:
                    raise InfrastructureError("Missing staged artifact")
                db.execute("INSERT OR IGNORE INTO grants VALUES(?,?)", (run_id, ref))
            db.execute(
                "INSERT INTO commits VALUES(?,?,?,?,?)",
                (run_id, turn, json_text(state), json_text(response), json_text(event)),
            )
            db.execute("UPDATE runs SET state=? WHERE run_id=?", (json_text(state), run_id))
        return response

    def trace(self, run_id):
        with self.connection() as db:
            rows = db.execute(
                "SELECT turn,response,event,state FROM commits WHERE run_id=? ORDER BY turn", (run_id,)
            ).fetchall()
        return [
            {"turn": r[0], "response": json.loads(r[1]), "event": json.loads(r[2]), "state": json.loads(r[3])}
            for r in rows
        ]

    def replay(self, run_id):
        rows = self.trace(run_id)
        if not rows:
            return self.get(run_id)
        for index, row in enumerate(rows, 1):
            if row["turn"] != index or row["state"]["state_version"] != index:
                raise InfrastructureError("Non-contiguous committed event log")
            if digest(row["state"]) != row["event"]["state_hash"]:
                raise InfrastructureError("Committed state integrity failure")
            if digest(row["response"]) != row["event"]["response_hash"]:
                raise InfrastructureError("Committed response integrity failure")
        return rows[-1]["state"]

    def cached_model(self, key, value=None):
        with self.connection() as db:
            if value is not None:
                db.execute("INSERT OR IGNORE INTO model_cache VALUES(?,?)", (key, json_text(value)))
            row = db.execute("SELECT value FROM model_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def export(self, run_id, include_raw=False):
        result = {"config": self.config(run_id), "trace": self.trace(run_id), "state": self.get(run_id)}
        with self.connection() as db:
            blobs = db.execute(
                "SELECT hash,content FROM blobs JOIN grants USING(hash) WHERE run_id=? ORDER BY hash",
                (run_id,),
            ).fetchall()
            audit = db.execute(
                "SELECT kind,payload FROM audit WHERE run_id=? ORDER BY seq", (run_id,)
            ).fetchall()
        result["artifacts"] = {ref: json.loads(content) for ref, content in blobs}
        result["operational_log"] = [{"kind": kind, **json.loads(payload)} for kind, payload in audit]
        if include_raw:
            with self.connection() as db:
                rows = db.execute(
                    "SELECT id,turn,request,response,complete,error FROM attempts WHERE run_id=? ORDER BY rowid",
                    (run_id,),
                ).fetchall()
            result["provider_archive"] = [
                dict(zip(["id", "turn", "request", "response", "complete", "error"], row)) for row in rows
            ]
        return result
