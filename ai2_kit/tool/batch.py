import os
import sys
import shutil
import shlex
from typing import Optional

from ai2_kit.core.util import ensure_dir, expand_globs, list_split
from ai2_kit.core.log import get_logger

logger = get_logger(__name__)


class BatchTool:
    """
    A toolkit to help generate batch scripts.
    """

    def run_cmd(self, *work_dirs: str, cmd: str):
        """
        Run command in each work directory.

        :param work_dirs: path or glob of work directories
        :param cmd: command to run, use {work_dir} to represent the work directory
        """
        paths = expand_globs(work_dirs)
        for path in paths:
            assert os.path.isdir(path), f'{path} is not a directory'
            _cmd = f"cd {shlex.quote(path)} && {cmd.format(work_dir=path)}"
            os.system(_cmd)
            logger.info(f'Run command: {_cmd}')
        return self

    def map_path(self, *sources: str, target: str, copy = False):
        """
        Map source files or directory to target path, use link by default.

        :param sources: path or glob of source files or directories
        :param target: target path, use {i} to represent the index of the source path,
        or {basename} to represent the basename of the source path
        :param copy: use copy instead of link
        """
        paths = expand_globs(sources)
        for i, path in enumerate(paths):
            target_path = target.format(i=i, basename=os.path.basename(path))
            ensure_dir(target_path)
            if copy:
                if os.path.isdir(path):
                    shutil.copytree(path, target_path)
                    logger.info(f'Copy directory {path} to {target_path}')
                else:
                    shutil.copy(path, target_path)
                    logger.info(f'Copy file {path} to {target_path}')
            else:
                path = os.path.abspath(path)
                os.symlink(path, target_path)
                logger.info(f'Link {path} to {target_path}')
        return self

    def gen_batches(self, *work_dirs: str,
                    out: str,
                    cmd: Optional[str]=None,
                    concurrency: int = 1,
                    header_file: Optional[str]=None,
                    suppress_error: bool = False,
                    checkpoint: bool = True,
                    checkpoint_file: str = 'done.ckpt',
                    rel_path: bool = False,
                    ):
        """
        Generate batch scripts for each work directory.

        This command will apply `cmd` to each work directory and generate batch scripts according to `concurrency`.

        :param work_dirs: path or glob of work directories
        :param out: path to write batch scripts, use {i} to represent the index of concurrent job
        :param cmd: command to run, if None, will read from stdin, use {word_dir} to represent the word directory,
        use {i} to represent the index of concurrent job
        :param concurrency: number of concurrent jobs, decide the number of batch scripts to generate, if 0, will generate one batch script for each work directory
        :param header_file: path to header file, will be added to the beginning of each batch script
        :param suppress_error: if True, will add `set -e` to the beginning of each batch script
        :param checkpoint: if True, will add checkpoint to each batch script, and skip the work directory if checkpoint exists
        :param checkpoint_file: checkpoint file name
        :param rel_path: if True, will use relative path in batch script
        """
        _work_dirs = expand_globs(work_dirs)
        # read cmd
        if cmd is None:
            cmd = sys.stdin.read()

        # read template
        header = '#!/bin/bash'
        if header_file is not None:
            with open(header_file, encoding='utf-8') as f:
                header = f.read()

        # generate batch scripts
        if concurrency <= 0:
            concurrency = len(_work_dirs)

        for i, job_group in enumerate(list_split(_work_dirs, concurrency)):
            batch = [ header ]
            if not suppress_error:
                batch.append('set -e')

            batch.append("for work_dir in \\")
            # generate batch script
            for work_dir in job_group:
                assert os.path.isdir(work_dir), f'{work_dir} is not a directory'
                if not rel_path:
                    work_dir = os.path.abspath(work_dir)
                batch.append(f'  {shlex.quote(work_dir)} \\')
            batch.extend([
                f'  ; do',
                f'    pushd $work_dir',
            ])
            if checkpoint:
                batch.extend([
                    f'    if [ -f {checkpoint_file} ]; then',
                    f'      echo "hit checkpoint, skip"',
                    f'      continue',
                    f'    fi',
                ])
            batch.extend([
                f'    {cmd}',
                f'    touch {checkpoint_file}',
                f'    popd',
                f'  done'
            ])

            # write batch script
            out_path = out.format(i=i)
            ensure_dir(out_path)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(batch))
            logger.info(f'Write batch script to {out_path}')
        return self

    def __str__(self) -> str:
        # suppress fire help message
        return ''

