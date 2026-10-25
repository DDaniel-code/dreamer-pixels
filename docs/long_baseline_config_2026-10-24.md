# Seed-0 Long-Baseline Configuration Audit

Status: historical pre-run audit. The configuration below was fixed before the
overnight run. The actual run later used this route, stopped at its wall-clock
limit, and reached 330,732 environment steps; see
`../OVERNIGHT_RESULT.md` for the verified result. This file remains an audit
of the intended configuration and is not itself a training log.

## Effective configuration

| Item | Effective value | Evidence / interpretation |
|---|---:|---|
| Task and observation | `cartpole_swingup`, camera 0, 64x64 RGB | Runner and 100k artifact config |
| Training seed | 0 | `config.json` |
| Action repeat | 2 | One decision advances two environment steps |
| Episode limit | 500 decision steps / 1,000 environment steps | Wrapper configuration |
| Prefill | 1,000 decision steps / 2,000 environment steps | First optimizer update is scheduled at decision step 1,000 |
| Batch and sequence | batch size 16, sequence length 50 | Preserved from the validated 100k run |
| Collection/update schedule | 1,000 decision steps, then 100 optimizer updates | `train_every=1000`, `train_steps=100`; not silently changed for the long run |
| Imagination horizon | 15 | Passed explicitly to the upstream algorithm |
| Replay capacity | 60,000 decision transitions, approximately 120,000 environment-step equivalents | Explicit RAM-budget setting; estimated raw RGB observation storage is about 703 MiB before replay metadata and allocator overhead |
| Precision and diagnostics | float32; anomaly detection off; nested progress bars off | Formal-run setting |
| Exploration | additive Gaussian training noise 0.3; evaluation noise 0 | `DMCDreamerAgent` wrapper; evaluation still samples latent posterior states, so `eval_noise=0` is not a claim of complete determinism |
| Long-run target | at most 250,000 decision steps / 500,000 environment steps | Wall-clock bounded; actual completion may be lower |
| Checkpoints | at 50k, 100k, 150k, and 250k decision steps when reached, plus latest-before-shutdown | Each checkpoint is an independent artifact in the long-run directory |

The runner treats one iteration as one decision step. Therefore:

```text
environment steps = 2 * decision steps
optimizer updates = 100 * number of scheduled update iterations
```

The old 100k run ended at 50,000 decision steps and 4,900 optimizer updates,
which is consistent with 1,000-step prefill followed by 49 scheduled update
iterations. Its training return is a diagnostic from the rlpyt logging path,
not the ten-episode evaluation curve.

## Paper and upstream comparison

The paper's Appendix A specifies batches of 50 sequences of length 50, Adam
learning rates of `6e-4` for the world model and `8e-5` for both actor and
value, imagination horizon 15, and a loop that initializes five random
episodes before collecting one episode for each block of 100 training updates.
It also describes a 30-dimensional stochastic state and 200-dimensional
deterministic state, and does not use latent overshooting, an entropy bonus, or
target value networks. See the [Dreamer paper Appendix A](https://arxiv.org/html/1912.01603v3).

The fixed PyTorch source at commit
`ccea6ae4a397a94c328891bd78e81d52dd156cb6` has these relevant defaults:

- batch size 50, sequence length 50, `train_every=1000`, `train_steps=100`;
- model, value, and actor learning rates `6e-4`, `8e-5`, and `8e-5`;
- horizon 15, discount 0.99, lambda 0.95, free nats 3, KL scale 1;
- replay size `5e6`, prefill 5,000 decision steps, and `use_pcont=False`;
- stochastic state 30, deterministic state 200, and hidden size 200 in the
  RSSM; the checked source uses 200-unit actor/value hidden layers and a
  300-unit reward model hidden layer.

The validated project run intentionally differs in a few recorded ways:

1. Batch size is 16 instead of the upstream/paper 50 to fit the available
   Kaggle GPU memory.
2. Prefill is 1,000 decision steps instead of the upstream default 5,000 and
   the paper's five random episodes.
3. The project wrapper collects 1,000 decision steps before each block of 100
   updates. The upstream `train_every` and `train_steps` values are retained,
   but the prefill and explicit wrapper contract are project choices.
4. Replay capacity is 60,000 transitions instead of the upstream default five
   million; this is a RAM-bounded engineering choice, not a claim that the
   paper used this capacity.
5. The long-run budget is at most 500,000 environment steps, far below the
   paper's multi-million-step evaluation horizon. It is evidence about this
   implementation and budget, not a paper-level reproduction.

`use_pcont` is false in the actual project run. The world-model loss therefore
does not include a continuation-prediction term; the upstream loss code uses a
zero placeholder for `pcont_loss`. This should not be confused with the
optional component existing in the generic upstream API.

## Resume classification

The 100k snapshot contains model parameters and the three Adam optimizer state
dictionaries. It does not contain replay contents, replay sampling state,
Python/NumPy/PyTorch RNG state, or an environment and recurrent-state boundary.
Consequently it is **B: model-and-optimizer-only continuation**, not a complete
resume. The 500k notebook is configured as **C: an independent seed-0 run**;
the old 100k pilot remains a separate result and will not be concatenated with
the new curve.

## Prepared entry point

The pre-run notebook was
`notebooks/kaggle_dreamer_v1_seed0_long_500k_20261024.ipynb`. The executed
overnight notebook was
`notebooks/kaggle_dreamer_v1_seed0_overnight_20261024.ipynb`; it wrote its
run-specific output under `/kaggle/working/dreamer_pytorch_pilot/` and used the
fixed Julius Frost source and rlpyt commits recorded above.
