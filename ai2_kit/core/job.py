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
    def is_success(self) -> bool:
        ...

    @abstractmethod
    def resubmit(self) -> 'JobFuture':
        ...

async def gather_jobs(jobs: List[JobFuture], timeout = float('inf'), max_tries: int = 1, raise_error=True) -> List[JobState]:
    async def wait_job(job: JobFuture) -> JobState:
        state = JobState.UNKNOWN
        tries = 0
        while True:
            try:
                state = await job.result_async(timeout)
                if state is JobState.COMPLETED:
                    return state
            except TimeoutError:
                state = JobState.TIMEOUT
            tries += 1

            if tries >= max_tries:
                break
            job = job.resubmit()

        if raise_error:
            raise RuntimeError(f'Job {job} failed with state {state}')
        else:
            return state

    return await asyncio.gather(*[wait_job(job) for job in jobs])
