from .pydantic import BaseModel
from typing import Optional, Mapping, TypedDict
import os
import copy


class ArtifactDict(TypedDict):
    """
    A dict representation of Artifact.
    Use this when you need to run a remote function call as pydantic model is not pickleable.

    referrer is not included in this dict as it is not pickleable.
    """
    url: str
    attrs: dict
    executor: Optional[str]
    format: Optional[str]
    includes: Optional[str]
    key: Optional[str]


class Artifact(BaseModel):

    key: Optional[str] = None
    executor: Optional[str] = None
    url: str
    format: Optional[str] = None
    includes: Optional[str] = None
    attrs: dict = dict()

    @classmethod
    def of(cls,
           url: str,
           key: Optional[str] = None,
           executor: Optional[str] = None,
           includes: Optional[str] = None,
           attrs: Optional[dict] = None,
           format: Optional[str] = None,):
        """
        Create an Artifact instance. Use this instead of __init__ to avoid type error of pydantic
        This method should be deprecated.
        """
        return cls(url=url,
                   executor=executor,
                   format=format,
                   includes=includes,
                   attrs=dict() if attrs is None else copy.deepcopy(attrs),  # deepcopy to reduce chance of mistake
                   key=key)

    def to_dict(self) -> ArtifactDict:
        """Convert to a dict representation.
        Use this when you need to run a remote function call as pydantic model is not pickleable."""
        return self.dict() # type: ignore

    def join(self, *paths, **kwargs):
        url = os.path.join(self.url, *paths)
        return Artifact.of(url=url, executor=self.executor, **kwargs)


ArtifactMap = Mapping[str, Artifact]
