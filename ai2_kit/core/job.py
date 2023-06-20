from typing import List, Callable, Optional, Awaitable
from enum import Enum
from abc import abstractmethod
import time
import asyncio

from .future import IFuture

# Copy from parsl
class JobState(bytes, Enum):
    """Defines a set of states that a job can be in"""
    def __new__(cls, value: int, terminal: bool, status_name: str) -> "JobState":
        obj = bytes.__new__(cls, [value])
        obj._value_ = value
        obj.terminal = terminal
        obj.status_name = status_name
        return obj

    value: int
    terminal: bool
    status_name: str

    UNKNOWN = (0, False, "UNKNOWN")
    PENDING = (1, False, "PENDING")
    RUNNING = (2, False, "RUNNING")
    CANCELLED = (3, True, "CANCELLED")
    COMPLETED = (4, True, "COMPLETED")
    FAILED = (5, True, "FAILED")
    TIMEOUT = (6, True, "TIMEOUT")
    HELD = (7, False, "HELD")

class TimeoutError(RuntimeError):
    ...

class JobFuture(IFuture[JobState]):

    @abstractmethod
    def get_job_state(self) -> JobState:
        ...

    @abstractmethod
    def cancel(self):
        ...

    @abstractmethod
    def redo(self):
        ...

    @abstractmethod
    def get_tries(self) -> int:
        ...

    @abstractmethod
    def is_success(self) -> bool:
        ...

    def is_retriable(self) -> bool:
        return True


DoneFn = Callable[[JobFuture], bool]

def _default_done_fn(_): return True

def retry_fn(max_tries=2) -> DoneFn:

    def callback(job: JobFuture):
        if job.is_success():
            return True
        if job.get_tries() >= max_tries:
            return True
        if job.is_retriable():
            job.redo()
            return False

    return callback  # type: ignore


async def gather_jobs(jobs: List[JobFuture], timeout = None, max_tries: int = 1) -> List[JobState]:
    async def wait_job(job: JobFuture) -> JobState:
        state = JobState.UNKNOWN
        tries = 0
        while True:
            try:
                state = await job.result_async(timeout)
                if state is JobState.COMPLETED:
                    break
            except TimeoutError:
                state = JobState.TIMEOUT
            tries += 1

            if tries >= max_tries:
                break
            job.redo()

        return state
    return await asyncio.gather(*[wait_job(job) for job in jobs])


# Deprecated
class GatherJobsFuture(IFuture[List[JobState]]):

    def __init__(self, jobs: List[JobFuture],
                 done_fn: DoneFn = _default_done_fn,
                 raise_exception = True,
                 polling_interval = 10,
                 ):
        self._jobs = jobs
        self._done_fn = done_fn
        self._raise_exception = raise_exception
        self._polling_interval = polling_interval

    def done(self):
        all_done = True
        for job in self._jobs:
            all_done = all_done and job.done() and self._done_fn(job)
        return all_done


    def result(self, timeout: Optional[float] = None):
        if timeout is None:
            timeout = float('inf')
        timeout_ts = time.time() + timeout

        while time.time() < timeout_ts:
            if self.done():
                if self._raise_exception:
                    failed_jobs = [
                        job for job in self._jobs if not job.is_success()]
                    if failed_jobs:
                        raise RuntimeError(
                            'Soem jobs are failed: {}'.format(failed_jobs))
                return [job.result() for job in self._jobs]

            else:
                time.sleep(self._polling_interval)
        else:
            raise TimeoutError('Wait for jobs timeout!')
