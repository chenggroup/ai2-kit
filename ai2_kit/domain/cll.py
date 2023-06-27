from abc import abstractmethod, ABC
from ai2_kit.core.artifact import Artifact, ArtifactMap
from ai2_kit.core.future import IFuture
from ai2_kit.core.resource_manager import ResourceManager
from typing import List, Callable, Any
from dataclasses import dataclass

@dataclass
class BaseCllContext:
    path_prefix: str
    resource_manager: ResourceManager

######################
# CLL Labeling Tasks #
######################


class ICllLabelOutput(ABC):

    @abstractmethod
    def get_labeled_system_dataset(self) -> List[Artifact]:
        ...

CllLabelTaskType = Callable[[Any, BaseCllContext], IFuture[ICllLabelOutput]]

######################
# CLL Training Tasks #
######################

class ICllTrainOutput(ABC):

    @abstractmethod
    def get_mlp_models(self) -> List[Artifact]:
        ...

    @abstractmethod
    def get_training_dataset(self) -> List[Artifact]:
        ...


CllTrainTaskType = Callable[[Any, BaseCllContext], IFuture[ICllTrainOutput]]

######################
# CLL Explore Tasks #
######################

class ICllExploreOutput(ABC):

    @abstractmethod
    def get_model_devi_dataset(self) -> List[Artifact]:
        ...

CllExploreTaskType = Callable[[Any, BaseCllContext], IFuture[ICllExploreOutput]]

######################
# CLL Select Task    #
######################

class ICllSelectorOutput(ABC):

    @abstractmethod
    def get_model_devi_dataset(self) -> List[Artifact]:
        ...

    @abstractmethod
    def get_passing_rate(self) -> float:
        ...

CllSelectorTaskType = Callable[[Any, BaseCllContext], IFuture[ICllSelectorOutput]]


def init_artifacts(artifacts: ArtifactMap):
    for key, artifact in artifacts.items():
        artifact.attrs.setdefault('ancestor', key)
