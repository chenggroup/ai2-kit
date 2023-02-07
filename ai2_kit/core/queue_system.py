from pydantic import BaseModel
from typing import Optional, Dict
from abc import ABC, abstractmethod
import shlex
import os
import re
import shortuuid
import time


from .connector import BaseConnector
from .log import get_logger
from .job import JobFuture, JobState

logger = get_logger(__name__)

class QueueSystemConfig(BaseModel):
    class Slurm(BaseModel):
        sbatch_bin: str = 'sbatch'
        squeue_bin: str = 'squeue'
        scancel_bin: str = 'scancel'
        polling_interval: int = 10

    class Lsf(BaseModel):
        bsub_bin: str = 'bsub'
        bjobs_bin: str = 'bjobs'
        polling_interval: int = 10

    slurm: Optional[Slurm]
    lsf: Optional[Lsf]


class BaseQueueSystem(ABC):

    connector: BaseConnector

    @abstractmethod
    def get_polling_interval(self) -> int:
        ...

    @abstractmethod
    def get_script_suffix(self) -> str:
        ...

    @abstractmethod
    def get_submit_cmd(self) -> str:
        ...

    @abstractmethod
    def get_job_id_pattern(self) -> str:
        ...

    @abstractmethod
    def get_job_state(self, job_id: str, success_indicator_path: str) -> JobState:
        ...

    @abstractmethod
    def cancel(self, job_id: str):
        ...

    def _post_submit(self, job: 'QueueJobFuture'):
        ...

    def submit(self, script: str, cwd: str, name: Optional[str] = None, success_indicator: Optional[str] = None):

        if name is None:
            name = shortuuid.uuid() + self.get_script_suffix()
        quoted_cwd = shlex.quote(cwd)

        if success_indicator is None:
            success_indicator = name + '.success'

        # create script
        script = '\n'.join([
            script,
            '',
            'touch {}'.format(shlex.quote(success_indicator)),
            '',
        ])

        path = os.path.join(cwd, name)
        self.connector.run('mkdir -p {}'.format(quoted_cwd))
        self.connector.dump_text(script, path)

        # submit script
        cmd = "cd {cwd} && {submit_cmd} {script}".format(
            cwd=quoted_cwd,
            submit_cmd=self.get_submit_cmd(),
            script=shlex.quote(name),
        )
        result = self.connector.run(cmd)

        m = re.search(self.get_job_id_pattern(), result.stdout)
        if m is None:
            raise RuntimeError("Unable to parse job id")
        job_id = m.group(1)
        job = QueueJobFuture(self,
                       job_id=job_id,
                       success_indicator=success_indicator,
                       name=name,
                       script=script,
                       cwd=cwd,
                       polling_interval=self.get_polling_interval() // 2,
                       )
        self._post_submit(job)
        return job


class Slrum(BaseQueueSystem):

    config: QueueSystemConfig.Slurm

    _last_states: Optional[Dict[str, JobState]]
    _last_update_at: float = 0

    translate_table = {
        'PD': JobState.PENDING,
        'R': JobState.RUNNING,
        'CA': JobState.CANCELLED,
        'CF': JobState.PENDING,  # (configuring),
        'CG': JobState.RUNNING,  # (completing),
        'CD': JobState.COMPLETED,
        'F': JobState.FAILED,  # (failed),
        'TO': JobState.TIMEOUT,  # (timeout),
        'NF': JobState.FAILED,  # (node failure),
        'RV': JobState.FAILED,  # (revoked) and
        'SE': JobState.FAILED   # (special exit state)
    }

    def get_polling_interval(self):
        return self.config.polling_interval

    def get_script_suffix(self):
        return '.sbatch'

    def get_submit_cmd(self):
        return self.config.sbatch_bin

    def get_job_id_pattern(self):
        # example: Submitted batch job 123
        return r"Submitted batch job\s+(\d+)"

    def get_job_state(self, job_id: str, success_indicator_path: str) -> JobState:
        state = self._get_all_states().get(job_id)
        if state is None:
            cmd = 'test -f {}'.format(shlex.quote(success_indicator_path))
            ret = self.connector.run(cmd, warn=True)
            if ret.return_code:
                return JobState.FAILED
            else:
                return JobState.COMPLETED
        else:
            return state

    def cancel(self, job_id: str):
        cmd = '{} {}'.format(self.config.scancel_bin, job_id)
        self.connector.run(cmd)

    def _post_submit(self, job: 'QueueJobFuture'):
        self._last_states = None

    def _translate_state(self, slurm_state: str) -> JobState:
        return self.translate_table.get(slurm_state, JobState.UNKNOWN)

    def _get_all_states(self) -> Dict[str, JobState]:
        current_ts = time.time()
        if self._last_states is not None and current_ts - self._last_update_at < self.config.polling_interval:
            return self._last_states

        cmd = "{} --noheader --format='%i %t' -u $(whoami)".format(
            self.config.squeue_bin)
        r = self.connector.run(cmd, hide=True)

        states: Dict[str, JobState] = dict()

        for line in r.stdout.split('\n'):
            if not line:  # skip empty line
                continue
            job_id, slurm_state = line.split()

            state = self._translate_state(slurm_state)
            states[job_id] = state
        self._last_update_at = current_ts
        self._last_states = states
        return states


class Lsf(BaseQueueSystem):

    config: QueueSystemConfig.Lsf

    def get_polling_interval(self):
        return self.config.polling_interval

    def get_script_suffix(self):
        return '.lsf'

    def get_submit_cmd(self):
        return self.config.bsub_bin + ' <'

    def get_job_id_pattern(self):
        # example: Job <123> is submitted to queue <small>.
        return r"Job <(\d+)> is submitted to queue"

    # TODO
    def get_job_state(self, job_id: str, success_indicator_path: str) -> JobState:
        return JobState.UNKNOWN

    # TODO
    def cancel(self, job_id: str):
        ...


class QueueJobFuture(JobFuture):

    def __init__(self,
                 queue_system: BaseQueueSystem,
                 job_id: str,
                 success_indicator: str,
                 script: str,
                 cwd: str,
                 name: str,
                 polling_interval=10,
                 ):
        self._queue_system = queue_system
        self._name = name
        self._script = script
        self._cwd = cwd
        self._job_id = job_id
        self._success_indicator = success_indicator
        self._polling_interval = polling_interval
        self._tries = 1
        self._done_state = None

    @property
    def success_indicator_path(self):
        return os.path.join(self._cwd, self._success_indicator)

    def get_job_state(self):
        if self._done_state is not None:
            return self._done_state

        state = self._queue_system.get_job_state(self._job_id, self.success_indicator_path)
        if state.terminal:
            self._done_state = state

        return state

    def redo(self):
        if not self.done():
            raise RuntimeError('Cannot redo an unfinished job!')
        job = self._queue_system.submit(
            script=self._script,
            cwd=self._cwd,
            name=self._name,
            success_indicator=self._success_indicator,
        )
        self._done_state = None
        self._tries += 1
        self._job_id = job._job_id

    def get_tries(self):
        return self._tries

    def is_success(self):
        return self.get_job_state() is JobState.COMPLETED

    def cancel(self):
        self._queue_system.cancel(self._job_id)

    def done(self):
        return self.get_job_state().terminal

    def result(self, timeout: float = float('inf')) -> JobState:

        timeout_ts = time.time() + timeout

        while time.time() < timeout_ts:
            if self.done():
                return self.get_job_state()
            else:
                time.sleep(self._polling_interval)
        else:
            raise RuntimeError(
                'Timeout of polling job: {}'.format(self._job_id))

    def __repr__(self):
        return repr(dict(
            name=self._name,
            cwd=self._cwd,
            job_id=self._job_id,
            success_indicator=self._success_indicator,
            polling_interval=self._polling_interval,
            state=self.get_job_state(),
        ))

