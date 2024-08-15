# Checkpoint

## Introduction
`ai2-kit` implements a checkpoint mechanism to allow user to resume a workflow from a previous checkpoint. This mechanism is implemented by saving the state of the workflow into a checkpoint file. The checkpoint file is a pickle dump file that can be executed to restore the workflow state.

## Usage
Suppose there is a time consuming step in the workflow, and you hope to set a checkpoint for this step so that when you rerun the workflow again, this step can be skipped. You can use the following code to set a checkpoint for this step:

```python
from ai2_kit.core.checkpoint import set_checkpoint_dir, apply_checkpoint
import time

def a_time_consuming_step(timeout):
    time.sleep(timeout)

# enable the checkpoint mechanism by setting a checkpoint directory
set_checkpoint_dir("./checkpoint")

# instead of calling the time consuming step directly,
# `a_time_consuming_step(10)`
# You should apply the checkpoint to the time consuming step
apply_checkpoint('time_consuming_step')(a_time_consuming_step)(10)

# The first time you run this script, it will take 10 seconds to finish.
# The second time you run this script, it will skip the time consuming step and finish immediately.
```

You can find some real world examples of using the checkpoint mechanism in the following source files:

* [ai2_kit/workflow/fep_mlp.py](../../ai2_kit/workflow/fep_mlp.py)
