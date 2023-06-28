from pydantic import BaseModel
from typing import Optional, Dict
from abc import ABC, abstractmethod
import shlex
import os
import re
import time
import asyncio


from .connector import BaseConnector
from .log import get_logger
from .job import JobFuture, JobState
from .checkpoint import apply_checkpoint, del_checkpoint
from .util import short_hash

logger = get_logger(__name__)


class QueueSystemConfig(BaseModel):
    class Slurm(BaseModel):
        sbatch_bin: str = 'sbatch'
        squeue_bin: str = 'squeue'
        scancel_bin: str = 'scancel'
        polling_interval: int = 10

    class LSF(BaseModel):
        bsub_bin: str = 'bsub'
        bjobs_bin: str = 'bjobs'
        polling_interval: int = 10

    slurm: Optional[Slurm]
    lsf: Optional[LSF]


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

    def submit(self, script: str, cwd: str,
               name: Optional[str] = None,
               checkpoint_key: Optional[str] = None,
               success_indicator: Optional[str] = None,
               ):

        # use hash instead of uuid to ensure idempotence
        if name is None:
            name = 'job-' + short_hash(script) + self.get_script_suffix()
        quoted_cwd = shlex.quote(cwd)

        # a placeholder file that will be created when the script end without error
        if success_indicator is None:
            success_indicator = name + '.success'

        # create script
        script = '\n'.join([
            script,
            '',
            f'touch {shlex.quote(success_indicator)}',
            '',
        ])

        script_path = os.path.join(cwd, name)
        self.connector.run(f'mkdir -p {quoted_cwd}')
        self.connector.dump_text(script, script_path)

        # submit script
        cmd = f"cd {quoted_cwd} && {self.get_submit_cmd()} {shlex.quote(name)}"

        # apply checkpoint
        submit_cmd_fn = self._submit_cmd
        if checkpoint_key is not None:
            submit_cmd_fn = apply_checkpoint(checkpoint_key)(submit_cmd_fn)

        logger.info(f'Submit batch script: {script_path}')
        job_id = submit_cmd_fn(cmd)

        job = QueueJobFuture(self,
                             job_id=job_id,
                             name=name,
                             script=script,
                             cwd=cwd,
                             checkpoint_key=checkpoint_key,
                             success_indicator=success_indicator,
                             polling_interval=self.get_polling_interval() // 2,
                             )
        self._post_submit(job)
        return job

    def _submit_cmd(self, cmd: str):
        result = self.connector.run(cmd)
        m = re.search(self.get_job_id_pattern(), result.stdout)
        if m is None:
            raise RuntimeError("Unable to parse job id")
        return m.group(1)


class Slurm(BaseQueueSystem):
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

    config: QueueSystemConfig.LSF

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
                 script: str,
                 cwd: str,
                 name: str,
                 success_indicator: str,
                 checkpoint_key: Optional[str],
                 polling_interval=10,
                 ):
        self._queue_system = queue_system
        self._name = name
        self._script = script
        self._cwd = cwd
        self._job_id = job_id
        self._checkpoint_key = checkpoint_key
        self._success_indicator = success_indicator
        self._polling_interval = polling_interval
        self._final_state = None

    @property
    def success_indicator_path(self):
        return os.path.join(self._cwd, self._success_indicator)

    def get_job_state(self):
        if self._final_state is not None:
            return self._final_state

        state = self._queue_system.get_job_state(
            self._job_id, self.success_indicator_path)
        if state.terminal:
            self._final_state = state

        return state

    def resubmit(self):
        if not self.done():
            raise RuntimeError('Cannot resubmit an unfinished job!')

        # delete checkpoint on resubmit
        if self._checkpoint_key is not None:
            del_checkpoint(self._checkpoint_key)

        return self._queue_system.submit(
            script=self._script,
            cwd=self._cwd,
            name=self._name,
            checkpoint_key=self._checkpoint_key,
            success_indicator=self._success_indicator,
        )

    def is_success(self):
        return self.get_job_state() is JobState.COMPLETED

    def cancel(self):
        self._queue_system.cancel(self._job_id)

    def done(self):
        return self.get_job_state().terminal

    def result(self, timeout: float = float('inf')) -> JobState:
        return asyncio.run(self.result_async(timeout))

    async def result_async(self, timeout: float = float('inf')) -> JobState:
        '''
        Though this is not fully async, as the job submission and state polling are still blocking,
        but it is already good enough to handle thousands of jobs (I guess).
        '''
        timeout_ts = time.time() + timeout
        while time.time() < timeout_ts:
            if self.done():
                return self.get_job_state()
            else:
                await asyncio.sleep(self._polling_interval)
        else:
            raise RuntimeError(f'Timeout of polling job: {self._job_id}')

    def __repr__(self):
        return repr(dict(
            name=self._name,
            cwd=self._cwd,
            job_id=self._job_id,
            success_indicator=self._success_indicator,
            polling_interval=self._polling_interval,
            state=self.get_job_state(),
        ))
