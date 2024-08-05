from typing import Optional, Dict
from abc import ABC, abstractmethod
from collections import defaultdict
import invoke
import shlex
import os
import re
import time
import asyncio
import json


from .connector import BaseConnector
from .log import get_logger
from .job import JobFuture, JobState
from .util import short_hash
from .pydantic import BaseModel

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

    class PBS(BaseModel):
        qsub_bin: str = 'qsub'
        qstat_bin: str = 'qstat'
        qdel_bin: str = 'qdel'

    slurm: Optional[Slurm] = None
    lsf: Optional[LSF] = None
    pbs: Optional[PBS] = None


class BaseQueueSystem(ABC):

    connector: BaseConnector

    def get_polling_interval(self) -> int:
        return 10

    def get_setup_script(self) -> str:
        return ''

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
    def get_job_id_envvar(self) -> str:
        ...

    @abstractmethod
    def get_job_state(self, job_id: str, success_indicator_path: str) -> JobState:
        ...

    @abstractmethod
    def cancel(self, job_id: str):
        ...

    def _post_submit(self, job: 'QueueJobFuture'):
        ...

    def submit(self,
               script: str,
               cwd: str,
               name: Optional[str] = None,
               success_indicator: Optional[str] = None,
               ):

        # use hash instead of uuid to ensure idempotence
        if name is None:
            name = 'job-' + short_hash(script) + self.get_script_suffix()
        quoted_cwd = shlex.quote(cwd)

        # a placeholder file that will be created when the script end without error
        if success_indicator is None:
            success_indicator = name + '.success'
        running_indicator = name + '.running'

        inject_cmds = '\n'.join([
            self.get_setup_script(),
            '',
        ])
        script = inject_cmd_to_script(script, inject_cmds)

        # create script and add a command to write job id to success indicator
        script = '\n'.join([
            script,
            '',
            f'echo ${self.get_job_id_envvar()} > {shlex.quote(success_indicator)}',
            '',
        ])

        script_path = os.path.join(cwd, name)
        self.connector.run(f'mkdir -p {quoted_cwd}')
        self.connector.dump_text(script, script_path)

        # submit script
        cmd = f"cd {quoted_cwd} && {self.get_submit_cmd()} {shlex.quote(name)}"

        # apply checkpoint
        submit_cmd_fn = self._submit_cmd

        # recover running job id
        # TODO: refactor the following code as function
        job_id, job_state  = None, JobState.UNKNOWN
        recover_cmd = f"cd {quoted_cwd} && cat {shlex.quote(running_indicator)}"
        try:
            job_id = self.connector.run(recover_cmd, hide=True).stdout.strip()
            if job_id:
                success_indicator_path = os.path.join(cwd, success_indicator)
                job_state = self.get_job_state(job_id, success_indicator_path=success_indicator_path)
        except:
            pass

        if job_id and job_state in (JobState.PENDING, JobState.RUNNING, JobState.COMPLETED):
            logger.info(f"{script_path} has been submmited ({job_id}) and in {str(job_state)} state, continue!")
        else:
            logger.info(f'Submit batch script: {script_path}')
            job_id = submit_cmd_fn(cmd)
            # create running indicator
            self.connector.dump_text(str(job_id), os.path.join(cwd, running_indicator))

        job = QueueJobFuture(self,
                             job_id=job_id,
                             name=name,
                             script=script,
                             cwd=cwd,
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

    _last_states = defaultdict(lambda: JobState.UNKNOWN)
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

    def get_job_id_envvar(self) -> str:
        return 'SLURM_JOB_ID'

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
        cmd = f'{self.config.scancel_bin} {job_id}'
        self.connector.run(cmd)

    def _post_submit(self, job: 'QueueJobFuture'):
        self._last_update_at = 0

    def _translate_state(self, slurm_state: str) -> JobState:
        return self.translate_table.get(slurm_state, JobState.UNKNOWN)

    def _get_all_states(self) -> Dict[str, JobState]:
        current_ts = time.time()
        if  (current_ts - self._last_update_at) < self.get_polling_interval():
            return self._last_states

        # call squeue to get all states
        cmd = f"{self.config.squeue_bin} --noheader --format='%i %t' -u $USER"
        try:
            r = self.connector.run(cmd, hide=True)
        except invoke.exceptions.UnexpectedExit as e:
            logger.warning(f'Error when calling squeue: {e}')
            return self._last_states

        states: Dict[str, JobState] = dict()
        for line in r.stdout.splitlines():
            if not line:  # skip empty line
                continue
            job_id, slurm_state = line.split()
            state = self._translate_state(slurm_state)
            states[job_id] = state
        # update cache
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

    def get_job_id_envvar(self) -> str:
        return 'LSB_JOBID'

    # TODO
    def get_job_state(self, job_id: str, success_indicator_path: str) -> JobState:
        return JobState.UNKNOWN

    # TODO
    def cancel(self, job_id: str):
        ...

    def _get_all_states(self) -> Dict[str, JobState]:
        ...


class PBS(BaseQueueSystem):
    config: QueueSystemConfig.PBS
    translate_table = {
        'B': JobState.RUNNING,  # This state is returned for running array jobs
        'R': JobState.RUNNING,
        'C': JobState.COMPLETED,  # Completed after having run
        'E': JobState.COMPLETED,  # Exiting after having run
        'H': JobState.HELD,  # Held
        'Q': JobState.PENDING,  # Queued, and eligible to run
        'W': JobState.PENDING,  # Job is waiting for it's execution time (-a option) to be reached
        'S': JobState.HELD  # Suspended
    }

    _last_states = defaultdict(lambda: JobState.UNKNOWN)
    _last_update_at: float = 0

    def get_setup_script(self) -> str:
        return 'cd $PBS_O_WORKDIR'

    def get_script_suffix(self) -> str:
        return '.pbs'

    def get_submit_cmd(self) -> str:
        return self.config.qsub_bin

    def get_job_id_pattern(self) -> str:
        return r"(.+)"

    def get_job_id_envvar(self) -> str:
        return 'PBS_JOBID'

    def cancel(self, job_id: str):
        cmd = f'{self.config.qdel_bin} {job_id}'
        self.connector.run(cmd)

    def _post_submit(self, job: 'QueueJobFuture'):
        self._last_update_at = 0  # force update stats

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

    def _get_all_states(self) -> Dict[str, JobState]:
        current_ts = time.time()
        if (current_ts - self._last_update_at) < self.get_polling_interval():
            return self._last_states

        cmd = f"{self.config.qstat_bin} -f -F json"
        try:
            r = self.connector.run(cmd, hide=True)
        except invoke.exceptions.UnexpectedExit as e:
            logger.warning(f'Error when calling qstat: {e}')
            return self._last_states

        states: Dict[str, JobState] = dict()
        qstat_json = json.loads(r.stdout)
        for job_id, job in qstat_json.get('Jobs', dict()).items():
            states[job_id] = self._translate_state(job['job_state'])
        self._last_states = states
        self._last_update_at = current_ts
        return states

    def _translate_state(self, slurm_state: str) -> JobState:
        return self.translate_table.get(slurm_state, JobState.UNKNOWN)


class QueueJobFuture(JobFuture):

    def __init__(self,
                 queue_system: BaseQueueSystem,
                 job_id: str,
                 script: str,
                 cwd: str,
                 name: str,
                 success_indicator: str,
                 polling_interval=10,
                 ):
        self._queue_system = queue_system
        self._name = name
        self._script = script
        self._cwd = cwd
        self._job_id = job_id
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

        logger.info(f'Resubmit job: {self._job_id}')
        return self._queue_system.submit(
            script=self._script,
            cwd=self._cwd,
            name=self._name,
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


def inject_cmd_to_script(script: str, cmd: str):
    """
    Find the position of first none comment or empty lines,
    and inject command before it
    """
    lines = script.splitlines()
    i = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith('#'):
            break
    lines.insert(i, cmd)
    return '\n'.join(lines)
