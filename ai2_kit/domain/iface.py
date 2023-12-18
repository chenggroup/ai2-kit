# This is a class to defined common interface

from abc import abstractmethod, ABC
from ai2_kit.core.artifact import Artifact, ArtifactMap
from ai2_kit.core.resource_manager import ResourceManager
from typing import List, Literal
from dataclasses import dataclass


TRAINING_MODE = Literal['default', 'dpff', 'fep-redox', 'fep-pka']


@dataclass
class BaseCllContext:
    path_prefix: str
    resource_manager: ResourceManager


class ICllLabelOutput(ABC):
    @abstractmethod
    def get_labeled_system_dataset(self) -> List[Artifact]:
        ...

class ICllTrainOutput(ABC):
    @abstractmethod
    def get_mlp_models(self) -> List[Artifact]:
        ...

    @abstractmethod
    def get_training_dataset(self) -> List[Artifact]:
        ...

class ICllExploreOutput(ABC):
    @abstractmethod
    def get_model_devi_dataset(self) -> List[Artifact]:
        ...

class ICllSelectorOutput(ABC):
    @abstractmethod
    def get_model_devi_dataset(self) -> List[Artifact]:
        ...

    @abstractmethod
    def get_new_explore_systems(self) -> List[Artifact]:
        ...

    @abstractmethod
    def get_passing_rate(self) -> float:
        ...


def init_artifacts(artifacts: ArtifactMap):
    for key, artifact in artifacts.items():
        artifact.attrs.setdefault('ancestor', key)
