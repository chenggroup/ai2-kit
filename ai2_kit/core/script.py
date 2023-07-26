from pydantic import BaseModel
from typing import Optional, List, Union, Sequence
import shlex


def exit_on_error_statment(v='__EXITCODE__'):
    return f"{v}=$?; if [ ${v} -ne 0 ]; then exit ${v}; fi"


def eoe_step(cmd: str):
    """
    exit on error step
    """
    return '\n'.join([cmd, exit_on_error_statment()])


class BashTemplate(BaseModel):
    shebang: str = '#!/bin/bash'
    header: str = ''
    setup: str = ''
    teardown: str = ''


class BashStep(BaseModel):
    cmd: Union[str, List[str]]
    cwd: Optional[str] = None
    checkpoint: Optional[str] = None
    exit_on_error: bool = True

    def render(self):
        if isinstance(self.cmd, list):
            self.cmd = ' '.join(self.cmd)

        rendered_step = '\n'.join([
            '#' * 80,
            self.cmd,
            '#' * 80,
        ])

        if self.exit_on_error:
            rendered_step = '\n'.join([
                rendered_step,
                exit_on_error_statment()
            ])

        if self.checkpoint:
            checkpoint = shlex.quote(self.checkpoint + '.checkpoint')
            msg = shlex.quote(f"hit {checkpoint}, skip")

            rendered_step = '\n'.join([
                f'if [ -f {checkpoint} ]; then echo {msg}; else',
                rendered_step,
                f'touch {checkpoint}; fi  # create checkpoint on success',
            ])

        if self.cwd:
            cwd = shlex.quote(self.cwd)
            rendered_step = '\n'.join([
                f'pushd {cwd} || exit 1',
                rendered_step,
                'popd',
            ])

        return rendered_step


BashSteps = Sequence[Union[str, BashStep]]


class BashScript(BaseModel):
    template: Optional[BashTemplate]
    steps: BashSteps

    def render(self):
        if self.template is None:
            return _render_bash_steps(self.steps)

        return '\n'.join([
            self.template.shebang,
            self.template.header,
            self.template.setup,
            _render_bash_steps(self.steps),
            self.template.teardown,
        ])


def _render_bash_steps(steps: BashSteps):
    rendered_steps = []

    for step in steps:
        if isinstance(step, str):
            rendered_steps.append(step)
        else:
            assert isinstance(step, BashStep)
            rendered_steps.append(step.render())
    return '\n\n'.join(rendered_steps)
