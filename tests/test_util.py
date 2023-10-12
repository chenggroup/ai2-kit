from ai2_kit.core.queue_system import inject_cmd_to_script
from unittest import TestCase

SLURM_SCRIPT_HEADER = """\
#!/bin/bash

#SBATCH -N 1
#SBATCH --partition cpu

#SBATCH --name

"""

class TestUtil(TestCase):


    def test_inject_cmd_to_script(self):
        cmd = "echo $SLUMR_JOB_ID > hello.running"
        in_script = '\n'.join([
            SLURM_SCRIPT_HEADER,
            'echo hello',
        ])
        expect_out = '\n'.join([
            SLURM_SCRIPT_HEADER,
            cmd,
            'echo hello',
        ])
        out_script = inject_cmd_to_script(in_script, cmd)
        self.assertEqual(out_script, expect_out)
