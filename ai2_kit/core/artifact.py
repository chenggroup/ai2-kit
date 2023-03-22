from pydantic import BaseModel
from typing import Optional, Mapping, TypedDict
import os
import copy


def __ArtifactDict():
    class ArtifactDict(TypedDict):
        """
        A dict representation of Artifact.
        Use this when you need to run a remote function call as pydantic model is not pickleable.

        referrer is not included in this dict as it is not pickleable.
        """
        executor: Optional[str]
        url: str
        format: Optional[str]
        includes: Optional[str]
        attrs: dict
    return ArtifactDict
ArtifactDict = __ArtifactDict()


class Artifact(BaseModel):
    executor: Optional[str]
    url: str
    format: Optional[str]
    includes: Optional[str]
    attrs: dict = dict()
    referrer: Optional['Artifact']

    @classmethod
    def of(cls,
           url: str,
           executor: Optional[str] = None,
           includes: Optional[str] = None,
           attrs: Optional[dict] = None,
           format: Optional[str] = None,
           referrer: Optional['Artifact'] = None):
        """Create an Artifact instance. Use this instead of __init__ to avoid type error of pydantic"""
        return cls(
            url=url,
            executor=executor,
            format=format,
            includes=includes,
            attrs=dict() if attrs is None else copy.deepcopy(attrs),  # deepcopy to reduce chance of mistake
            referrer=referrer,
        )

    def to_dict(self) -> ArtifactDict:
        """Convert to a dict representation.
        Use this when you need to run a remote function call as pydantic model is not pickleable."""
        return self.dict(exclude={'referrer',})  # type: ignore

    def join(self, *paths, **kwargs):
        url = os.path.join(self.url, *paths)
        return Artifact.of(url=url, executor=self.executor, referrer=self, **kwargs)


ArtifactMap = Mapping[str, Artifact]
