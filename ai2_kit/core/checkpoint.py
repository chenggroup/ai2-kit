from typing import TypeVar, Union, Callable, NamedTuple, Optional
from threading import Lock
import functools
import cloudpickle
import os
import inspect
import fnmatch

from .log import get_logger
from .util import to_awaitable

logger = get_logger(__name__)

_lock = Lock()
_checkpoint_file: Optional[str] = None
_checkpoint_data: Optional[dict] = None


class FnInfo(NamedTuple):
    fn_name: str
    args: tuple
    kwargs: dict
    call_site: str


KeyFn = Callable[[FnInfo], str]

EMPTY = object()


def set_checkpoint_file(path: str):
    global _checkpoint_file
    if _checkpoint_file is not None:
        raise RuntimeError(
            "checkpoint path has been set to {}".format(_checkpoint_file))
    _checkpoint_file = path
    _load_checkpoint()


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

            if disable or _checkpoint_file is None:
                return fn(*args, **kwargs)

            ret = _get_checkpoint(key)
            if ret is not EMPTY:
                return ret

            ret = fn(*args, **kwargs)
            if inspect.isawaitable(ret):
                async def _wrap_fn():
                    _ret = await ret
                    _set_checkpoint(key, _ret, fn_info, True)
                    return _ret
                return _wrap_fn()
            else:
                _set_checkpoint(key, ret, fn_info, False)
                return ret

        return wrapper # type: ignore
    return _checkpoint


def _load_checkpoint():
    global _checkpoint_data
    if _checkpoint_data is not None:
        return
    assert _checkpoint_file is not None, '_checkpoint_path should not be None!'
    if os.path.exists(_checkpoint_file):
        with open(_checkpoint_file, 'rb') as f:
            _checkpoint_data = cloudpickle.load(f)
    else:
        _checkpoint_data = dict()


def _dump_checkpoint():
    assert _checkpoint_data is not None, '_checkpoint_data should not be None!'
    with open(_checkpoint_file, 'wb') as f:  # type: ignore
        cloudpickle.dump(_checkpoint_data, f)


def _get_checkpoint(key: str):
    try:
        with _lock:
            _load_checkpoint()
            assert _checkpoint_data is not None
            value = _checkpoint_data.get(key, None)
            if value is None:
                return EMPTY
            logger.info(f"Hit checkpoint: {key}")
            if value['is_awaitable']:
                return to_awaitable(value['return'])
            else:
                return value['return']

    except Exception as e:
        logger.error(f"Fail to get checkpoint: {key}", e)
        return EMPTY


def _set_checkpoint(key: str, value, info: FnInfo, is_awaitable: bool = False):
    try:
        with _lock:
            assert _checkpoint_data is not None
            # args, kwargs may contain unpickable objects
            _checkpoint_data[key] = {
                'return': value,
                'is_awaitable': is_awaitable,
                'info': {
                    'fn_name': info.fn_name,
                    'call_site': info.call_site,
                }
            }
            _dump_checkpoint()
    except Exception as e:
        logger.error('Fail to set checkpoint', e)


def del_checkpoint(key: str):
    try:
        with _lock:
            _load_checkpoint()
            assert _checkpoint_data is not None
            if key in _checkpoint_data:
                del _checkpoint_data[key]
                _dump_checkpoint()
    except Exception as e:
        logger.error('Fail to delete checkpoint', e)


class CheckpointCmd:
    """checkpoint command line interface"""
    def load(self, file):
        set_checkpoint_file(file)
        return self

    def ls(self, verbose=False):
        '''list all the checkpoint entries in the checkpoint file'''
        assert _checkpoint_data is not None
        for i, (key, value) in enumerate(_checkpoint_data.items()):
            if verbose:
                print('\n'.join([
                    '=' * 80,
                    f'Key:        \t{key}',
                    f'Call Site:  \t{value["info"]["call_site"]}',
                    f'Function:   \t{value["info"]["fn_name"]}',
                ]))
            else:
                print(key)

    def rm(self, glob_pattern: str, yes=False, exclude: Optional[str]=None):
        """remove checkpoint entries with the given pattern"""
        assert _checkpoint_data is not None

        keys = [ key for key in _checkpoint_data.keys() if fnmatch.fnmatch(key, glob_pattern) ]
        if exclude is not None:
            keys = [ key for key in keys if not fnmatch.fnmatch(key, exclude) ]

        for key in keys:
            if not yes:
                print(f"Delete checkpoint {key}? [y/n]")
                if input().lower() != 'y':
                    continue
            del _checkpoint_data[key]
            print(f"Delete checkpoint {key}")
        _dump_checkpoint()
