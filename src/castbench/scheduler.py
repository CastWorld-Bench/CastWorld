import heapq
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import count
from threading import Event

from .common import InfrastructureError


def run_jobs(
    jobs,
    execute,
    max_workers=2,
    on_complete=None,
    stop=None,
    *,
    retry_if=None,
    max_retries=0,
    retry_delays=(0.0,),
    on_retry=None,
    on_retry_exhausted=None,
    job_label=None,
    continue_after_retry_exhaustion=False,
):
    """Run bounded jobs, optionally requeueing isolated transient failures in place."""
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    delays = tuple(float(value) for value in retry_delays)
    if not delays or any(value < 0 for value in delays):
        raise ValueError("retry_delays must contain nonnegative values")
    stop = stop if stop is not None else Event()
    iterator = iter(jobs)
    results, failures = [], []
    delayed = []
    sequence = count()

    def label(job):
        value = job_label(job) if job_label else repr(job)
        text = str(value).replace("\n", " ")
        return text if len(text) <= 240 else text[:237] + "..."

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = {}

        def fill():
            while len(pending) < max_workers and not stop.is_set():
                now = time.monotonic()
                if delayed and delayed[0][0] <= now:
                    _, _, job, retry_count = heapq.heappop(delayed)
                    pending[pool.submit(execute, job)] = (job, retry_count)
                    continue
                try:
                    job = next(iterator)
                except StopIteration:
                    break
                pending[pool.submit(execute, job)] = (job, 0)

        fill()
        while pending or delayed:
            if not pending:
                if stop.is_set():
                    break
                time.sleep(max(0.0, delayed[0][0] - time.monotonic()))
                fill()
                continue
            timeout = max(0.0, delayed[0][0] - time.monotonic()) if delayed else None
            done, _ = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                fill()
                continue
            for future in done:
                job, retry_count = pending.pop(future)
                try:
                    value = future.result()
                    results.append(value)
                    if on_complete:
                        on_complete(value)
                except Exception as exc:
                    retryable = retry_if is not None and retry_if(exc)
                    if retryable and retry_count < max_retries and not stop.is_set():
                        delay = delays[min(retry_count, len(delays) - 1)]
                        heapq.heappush(
                            delayed,
                            (time.monotonic() + delay, next(sequence), job, retry_count + 1),
                        )
                        if on_retry:
                            on_retry(job, exc, retry_count + 1, delay)
                    else:
                        failures.append((job, exc))
                        if retryable and on_retry_exhausted:
                            on_retry_exhausted(job, exc, retry_count)
                        if not (retryable and continue_after_retry_exhaustion):
                            stop.set()
            fill()
    if failures:
        job, exc = failures[0]
        suffix = f"; {len(failures)} job(s) failed" if len(failures) > 1 else ""
        raise InfrastructureError(
            f"Joint evaluation paused at {label(job)}: {type(exc).__name__}: {exc}{suffix}"
        ) from exc
    if stop.is_set():
        raise InfrastructureError("Joint evaluation stopped; completed work was preserved")
    return results


def run_many(runners, max_workers=1):
    """Parallelize independent Episodes only. A paused run is never converted to a low score."""
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    ids = [r.run_id for r in runners]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate logical run IDs")

    def execute(runner):
        try:
            state = runner.run()
            return {"run_id": runner.run_id, "status": "finished", "final": state["final"]}
        except InfrastructureError as exc:
            return {"run_id": runner.run_id, "status": "paused_infrastructure", "reason": str(exc)}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(execute, runners))
