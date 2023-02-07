from pydantic import BaseModel
from typing import Optional, Dict
import os


class Artifact(BaseModel):
    executor: Optional[str]
    url: str
    glob: Optional[str]
    attrs: dict = dict()

    referrer: Optional['Artifact']

    def join(self, *paths):
        url = os.path.join(self.url, *paths)
        return Artifact(
            executor=self.executor,
            url=url,
            referrer=self,
        ) # type: ignore

ArtifactMap = Dict[str, Artifact]
