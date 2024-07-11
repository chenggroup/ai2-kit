from ai2_kit.core.util import expand_globs
from ai2_kit.core.log import get_logger
from ai2_kit.core.job import JobState
from ai2_kit.core.cmd import CmdGroup

import datetime
import shlex
import time
import os
import re


logger = get_logger(__name__)


def append_if_not_exist(fp, line: str, feat_str = None):
    if feat_str is None:
        feat_str = line
    for line in fp:
        if feat_str in line:
            return
    fp.seek(0, os.SEEK_END)
    fp.write(f'\n{line}\n')


class Slurm:

    _state_table = {
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

    def __init__(self, sbatch_bin = 'sbatch', squeue_bin = 'squeue', scancel_bin='scancel') -> None:
        self._sbatch_bin = sbatch_bin
        self._squeue_bin = squeue_bin
        self._scancel_bin = scancel_bin
        self._job_states = {}

    def submit(self, *path_or_glob: str):
        """
        Submit multiple Slurm script at once.

        :param path_or_glob: path or glob of Slurm script
        """
        files = expand_globs(path_or_glob, raise_invalid=True)
        if not files:
            raise ValueError('No files found')
        for file in files:
            with open(file, '+a') as fp:
                append_if_not_exist(fp, 'touch slurm_$SLURM_JOB_ID.done # AUTO GENERATED')
        try:
            for file in files:
                job_id = self._submit(file)
                logger.info(f'Submitted job {job_id} for {file}')
                self._job_states[job_id] = JobState.UNKNOWN
        except:
            self._cancel()
            raise
        return self

    def wait(self, timeout=3600 * 24 * 7, ignore_error = False, fast_fail = False, interval = 10):
        """
        Wait until all jobs are finished.

        :param timeout: timeout in seconds
        :param ignore_error: ignore error
        :param fast_fail: exit if any job failed
        :param interval: interval in seconds
        """
        fail_cnt = 0
        start_at = datetime.datetime.now()
        while (datetime.datetime.now() - start_at).total_seconds() < timeout:
            try:
                self._update_job_states()
                logger.info('Job states: %s', self._job_states)
                fail_cnt = 0
            except Exception:
                fail_cnt += 1
                logger.exception('Failed to update job states')
                if fail_cnt > 5:  # stop if keep failing
                    raise
            if all(state.terminal for state in self._job_states.values()):
                break
            if fast_fail and self._is_any_failed():
                self._cancel()
                raise RuntimeError('Fast fail!')
            time.sleep(interval)
        else:
            logger.error('Timeout')
            if not ignore_error:
                raise RuntimeError('Timeout')
        if not ignore_error and self._is_any_failed():
            raise RuntimeError('Some jobs failed')

    def _is_any_failed(self):
        return any(state == JobState.FAILED for state in self._job_states.values())

    def _update_job_states(self):
        query_cmd = f"{self._squeue_bin} --noheader --format='%i %t' -u $USER"

        fp = os.popen(query_cmd)
        out = fp.read()
        exit_code = fp.close()
        if exit_code is not None:
            raise RuntimeError(f'Failed to query job states: {exit_code}')

        state = {}
        for line in out.splitlines():
            if line:
                job_id, slurm_state = line.split()
                state[job_id] = slurm_state
        for job_id in self._job_states:
            if job_id in state:
                self._job_states[job_id] = self._state_table[state[job_id]]
            else:
                if os.path.exists(f'slurm_{job_id}.done'):
                    self._job_states[job_id] = JobState.COMPLETED
                else:
                    self._job_states[job_id] = JobState.FAILED

    def _cancel(self):
        for job_id in self._job_states:
            os.system(f'{self._scancel_bin} {job_id}')

    def _submit(self, file: str):
        submit_cmd = f'{self._sbatch_bin} {shlex.quote(file)}'
        with os.popen(submit_cmd) as fp:
            stdout = fp.read()
        m = re.search(r'Submitted batch job (\d+)', stdout)
        if m is None:
            raise ValueError(f'Failed to submit job: {stdout}')
        job_id = m.group(1)
        return job_id

cmd_entry = CmdGroup(items={
    'slurm': Slurm,
}, doc='Tools to facilitate HPC related tasks.')
