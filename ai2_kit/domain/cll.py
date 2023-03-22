from abc import abstractmethod, ABC
from ai2_kit.core.artifact import Artifact
from ai2_kit.core.future import IFuture
from ai2_kit.core.resource_manager import ResourceManager
from typing import List, Callable, Optional
from dataclasses import dataclass

@dataclass
class BaseCllContext:
    path_prefix: str
    resource_manager: ResourceManager

######################
# CLL Labeling Tasks #
######################

class ICllLabelInput(ABC):

    @abstractmethod
    def set_systems(self, systems: List[Artifact]):
        ...


class ICllLabelOutput(ABC):

    @abstractmethod
    def get_labeled_system_dataset(self) -> List[Artifact]:
        ...

CllLabelTaskType = Callable[[ICllLabelInput, BaseCllContext], IFuture[ICllLabelOutput]]

######################
# CLL Training Tasks #
######################

class ICllTrainInput(ABC):

    @abstractmethod
    def update_training_dataset(self, data: List[Artifact]):
        ...


class ICllTrainOutput(ABC):

    @abstractmethod
    def get_mlp_models(self) -> List[Artifact]:
        ...

    @abstractmethod
    def get_training_dataset(self) -> List[Artifact]:
        ...


CllTrainTaskType = Callable[[ICllTrainInput, BaseCllContext], IFuture[ICllTrainOutput]]

######################
# CLL Explore Tasks #
######################

class ICllExploreInput(ABC):

    @abstractmethod
    def set_md_models(self, models: List[Artifact]):
        ...

    @abstractmethod
    def set_fep_models(self, red_models: List[Artifact], neu_models: List[Artifact]):
        ...


class ICllExploreOutput(ABC):

    @abstractmethod
    def get_model_devi_dataset(self) -> List[Artifact]:
        ...


CllExploreTaskType = Callable[[ICllExploreInput, BaseCllContext], IFuture[ICllExploreOutput]]

######################
# CLL Select Task    #
######################

class ICllSelectorInput(ABC):

    @abstractmethod
    def set_model_devi_dataset(self, data: List[Artifact]):
        ...

class ICllSelectorOutput(ABC):

    @abstractmethod
    def get_model_devi_dataset(self) -> List[Artifact]:
        ...

    @abstractmethod
    def get_passing_rate(self) -> float:
        ...

CllSelectorTaskType = Callable[[ICllSelectorInput, BaseCllContext], IFuture[ICllSelectorOutput]]

