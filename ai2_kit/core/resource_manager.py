from typing import Dict, List, Tuple, Union, Optional, Sequence
import copy

from .artifact import Artifact, ArtifactMap
from .executor import Executor, ExecutorMap, create_executor


ArtifactOrKey = Union[Artifact, str]

class ResourceManager:

    @property
    def default_executor(self):
        return self.get_executor()

    def __init__(self,
                 executor_configs: ExecutorMap,
                 artifacts: ArtifactMap,
                 default_executor: str,
                 ) -> None:

        self._executor_configs = executor_configs
        self._default_executor_name = default_executor
        self._executors: Dict[str, Executor] = dict()
        # runtime check to ensure quick failure
        self.default_executor

        # fill in default values
        for key, artifact in artifacts.items():
            if artifact.executor is None:
                artifact.executor = self.default_executor.name
            if artifact.key is None:
                artifact.key = key
        self._artifacts = artifacts

    def get_executor(self, name: Optional[str] = None) -> Executor:
        if name is None:
            name = self._default_executor_name

        config = self._executor_configs.get(name)
        if config is None:
            raise ValueError(
                'Executor with name {} is not found!'.format(name))

        if name not in self._executors:
            executor = create_executor(config, name)
            executor.init()
            self._executors[name] = executor

        return self._executors[name]

    def get_artifact(self, key: str) -> Artifact:
        # raise error it is by designed, ensure quick failure
        return self._artifacts[key]

    def get_artifacts(self, keys: List[str]) -> List[Artifact]:
        return [self.get_artifact(key) for key in keys]

    def resolve_artifact(self, artifact: ArtifactOrKey ) -> List[Artifact]:
        # TODO: support cross executor data resolve in the future
        if isinstance(artifact, str):
            artifact = self.get_artifact(artifact)
        paths = self.default_executor.resolve_artifact(artifact)

        result = [Artifact.of(
            url=path,
            format=artifact.format,
            includes=None,  # has been consumed
            attrs=copy.deepcopy(artifact.attrs),
            executor=self.default_executor.name,
        ) for path in paths]

        assert len(result) > 0, f'artifact {artifact} is invalid'
        return result

    def resolve_artifacts(self, artifacts: Sequence[ArtifactOrKey]) -> List[Artifact]:
        # flat map
        return [x for a in artifacts for x in self.resolve_artifact(a)]
