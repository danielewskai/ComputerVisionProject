# Voice access refactor

## Files

- `voice_access_pipeline.py` - all functions and classes.
- `voice_access_workflow.ipynb` - notebook with only calls to functions.

## Main idea

Notebook is split into four independent stages:

1. data preparation,
2. training,
3. evaluation of a saved model,
4. comparison of experiments.

## Main functions

- `prepare_data_artifacts(...)`
- `train_experiment(...)`
- `evaluate_saved_experiment(...)`
- `run_full_experiment(...)`
- `compare_experiments(...)`
- `clone_experiment(...)`

## Why this structure

- all logic is outside the notebook,
- training and evaluation are separated,
- experiments are easier to compare,
- saved checkpoints and csv results are reusable,
- adding a new allowed speaker is easier to support later.
