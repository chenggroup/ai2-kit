from typing import Generic, Optional, TypeVar, Callable, Awaitable
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

class DummyFuture(IFuture[T]):

    def __init__(self, value: T) -> None:
        self.value = value

    def done(self):
        return True

    def result(self, timeout: Optional[float] = None) -> T:
        return self.value


NT = TypeVar('NT')

MapFn = Callable[[T], NT]

class MapFuture(IFuture[NT]):

    def __init__(self, future: IFuture[T], map_fn: MapFn):
        self._future = future
        self._map_fn = map_fn

    def done(self):
        return self._future.done()

    def result(self, timeout=None) -> NT:
        result = self._future.result(timeout)
        return self._map_fn(result)

def map_future(future: IFuture, value: T):
    return MapFuture[T](future, lambda _ : value)