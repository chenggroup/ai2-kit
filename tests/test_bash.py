from ai2_kit.core.script import make_gpu_parallel_steps, _render_bash_steps
from unittest import TestCase
import tempfile
import os


class TestConfig(TestCase):

    def test_gpu_parallel_steps(self):

        step_groups = [
            [
                'sleep 3',
                'echo "group 1 stop"',
            ],
            [
                'sleep 3',
                'echo "group 2 stop"',
            ],
            [
                'sleep 3',
                'echo "group 3 stop"',
            ],
        ]
        p_steps = make_gpu_parallel_steps(step_groups)
        script = _render_bash_steps(p_steps)
        print(script)

        # write script to temp
        with tempfile.NamedTemporaryFile('w') as f:
            f.write('export CUDA_VISIBLE_DEVICES=0,1\n')
            f.write(script)
            f.flush()
            os.system(f'bash {f.name}')
