from typing import TypeVar, Union, Callable, NamedTuple
from threading import Lock
import functools
import cloudpickle
import os
import inspect

from .log import get_logger
from .util import to_awaitable, ensure_dir

logger = get_logger(__name__)


class FnInfo(NamedTuple):
    fn_name: str
    args: tuple
    kwargs: dict
    call_site: str

KeyFn = Callable[[FnInfo], str]
EMPTY = object()

class CheckpointService:

    def __init__(self):
        self.checkpoint_dir = None
        self._lock = Lock()

    def set_checkpoint_dir(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
        except FileExistsError:
            logger.error("ai2-kit use directory as checkpoint since 0.17.0")
            raise

    def apply_checkpoint(self, key_fn: Union[str, KeyFn], disable = False):
        call_site = inspect.getframeinfo(inspect.stack()[1][0])

        T = TypeVar('T', bound=Callable)
        def _checkpoint(fn: T) -> T:

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                fn_info = FnInfo(
                    fn_name=fn.__name__,
                    args=args,
                    kwargs=kwargs,
                    call_site=f'{call_site.filename}:{call_site.lineno}',
                )

                key = key_fn if isinstance(key_fn, str) else key_fn(fn_info)

                if disable or self.checkpoint_dir is None:
                    return fn(*args, **kwargs)

                ret = self._get_checkpoint(key)
                if ret is not EMPTY:
                    logger.info(f"Hit checkpoint: {key}")
                    return ret

                ret = fn(*args, **kwargs)
                if inspect.isawaitable(ret):
                    async def _wrap_fn():
                        _ret = await ret
                        self._set_checkpoint(key, _ret, fn_info, True)
                        return _ret
                    return _wrap_fn()
                else:
                    self._set_checkpoint(key, ret, fn_info, False)
                    return ret

            return wrapper # type: ignore
        return _checkpoint

    def _get_checkpoint(self, key: str):
        assert self.checkpoint_dir is not None
        try:
            with self._lock:
                checkpoint_file = os.path.join(self.checkpoint_dir, key)
                if not os.path.exists(checkpoint_file):
                    return EMPTY
                with open(checkpoint_file, 'rb') as f:
                    value = cloudpickle.load(f)
                if value['is_awaitable']:
                    return to_awaitable(value['return'])
                else:
                    return value['return']
        except Exception as e:
            logger.exception(f"Fail to get checkpoint: {key}")
            return EMPTY


    def _set_checkpoint(self, key: str, value, info: FnInfo, is_awaitable: bool = False):
        assert self.checkpoint_dir is not None
        try:
            with self._lock:
                # args, kwargs may contain unpickable objects
                _checkpoint_data = {
                    'return': value,
                    'is_awaitable': is_awaitable,
                    'info': {
                        'fn_name': info.fn_name,
                        'call_site': info.call_site,
                    }
                }
                checkpoint_file = os.path.join(self.checkpoint_dir, key)
                ensure_dir(checkpoint_file)
                with open(checkpoint_file, 'wb') as f:
                    cloudpickle.dump(_checkpoint_data, f)

        except Exception as e:
            logger.exception('Fail to set checkpoint')


checkpoint_service = CheckpointService()


def set_checkpoint_dir(checkpoint_dir: str):
    checkpoint_service.set_checkpoint_dir(checkpoint_dir)


def apply_checkpoint(key_fn: Union[str, KeyFn], disable = False):
    """
    apply checkpoint for function.

    Note: This checkpoint implementation doesn't support multiprocess.
    To support multiple process we need to have a dedicated background process to read/write checkpoint,
    which will require message queue (e.g. nanomsg or nng) to implement it.

    Example:

    >>> set_checkpoint_file('/tmp/test.ckpt')
    >>> task_fn = lambda a, b: a + b
    >>> checkpoint('task_1+2')(task_fn)(1, 2)
    """

    return checkpoint_service.apply_checkpoint(key_fn, disable)
