# Kaggle artifact rescue (2026-10-23)

## Scope

This rescue did not start a session or rerun training. It opened the saved
Kaggle Version 1, downloaded the notebook with executed outputs, and checked
the saved version's Output and Logs tabs.

Local copy:

`artifacts/kaggle/notebook0e96057666_version1/notebook0e96057666.ipynb`

The saved version URL was
`notebook0e96057666?scriptVersionId=361600149`. Kaggle reported `Output: 0
files`, so the working-directory binaries and raw logs were not persisted as
downloadable outputs after the session stopped.

## Debug evidence recovered from notebook output

The inventory cell recorded the following during the live session:

```text
debug_summary.json exists=True, bytes=480
eval_episodes.csv exists=True, bytes=861
summary.json exists=True, bytes=444
metrics.jsonl exists=True, bytes=282998
source_manifest.json exists=True, bytes=369
compatibility_patches.json exists=True, bytes=120
debug_checkpoint_exists=True
```

The original `eval_episodes.csv` rows were recovered from the notebook output
into `artifacts/kaggle/notebook0e96057666_version1/debug_eval_episodes.csv`.
They are explicitly labeled as the **10-update debug policy**. For seeds
1000--1009, with 500 decision steps and 1000 environment steps per episode:

| Policy | Mean return | Sample standard deviation |
| --- | ---: | ---: |
| Random | 42.727441783551065 | 20.45684927379362 |
| 10-update debug policy | 74.09244370294698 | 2.4774558691970197 |

The individual returns are in the recovered CSV. These are not final-training
results, not ten training seeds, and not a paper-reproduction claim.

## Training checkpoint interpretation

The live inventory cell reported `train_checkpoint_exists=True` and the
training log recorded:

```text
prefill=1000
itr #999 saving snapshot...
itr #999 saved
```

The pinned upstream algorithm returns before optimization when `itr <
prefill`. Therefore the `itr #999` snapshot is at the prefill boundary and
cannot be called an updated-policy checkpoint. The binary was not persisted by
Kaggle's saved version, so its parameter state cannot now be inspected.

The wrapper also reported:

```text
train_summary.json: absent
progress.csv: absent
metrics.jsonl: present during session, 282998 bytes
```

Because the raw metrics file was not persisted, no loss-versus-optimizer-update
plot is claimed or generated. Because no completed training return exists, no
return learning curve is claimed or generated.
