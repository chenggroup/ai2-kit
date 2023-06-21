from typing import Generic, Optional, TypeVar
from abc import abstractmethod


T = TypeVar('T')


class IFuture(Generic[T]):

    @abstractmethod
    def done(self) -> bool:
        pass

    @abstractmethod
    def result(self, timeout: Optional[float] = None) -> T:
        pass

    async def result_async(self, timeout: Optional[float] = None) -> T:
        ...