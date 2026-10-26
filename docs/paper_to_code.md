# Paper to code map

The active paths below correspond to `juliusfrost/dreamer-pytorch@ccea6ae4a397a94c328891bd78e81d52dd156cb6`, with `rlpyt@f04f23db1eb7b5915d88401fca67869968a07a37`. The TensorFlow repositories are historical references only. This repository does not copy upstream source code.

The upstream PyTorch repository is in maintenance mode and MIT-licensed. The
Kaggle pilot retrieves the source and submodule at these exact commits and
keeps their `LICENSE` files in the working checkout.

## Active PyTorch mapping

- RSSM prior/posterior and latent transition: `dreamer/models/rnns.py`
  (`RSSMTransition`, `RSSMRepresentation`, `RSSMRollout`).
- Image encoder/decoder: `dreamer/models/observation.py`.
- Reward and value models: `dreamer/models/dense.py`.
- Actor distribution and action decoder: `dreamer/models/action.py` and
  `dreamer/models/distribution.py`.
- Replay buffer and sequence sampling: `dreamer/algos/replay.py`, using the
  pinned `rlpyt` sequence replay implementation.
- World-model, actor, and critic losses plus latent imagination:
  `dreamer/algos/dreamer_algo.py`.
- DMC pixel environment, action repeat, normalization, and time limit:
  `dreamer/envs/dmc.py`, `action_repeat.py`, `normalize_actions.py`, and
  `time_limit.py`; the pilot adds only a seeded adapter around these paths.
- Sampling entry point: `main_dmc.py`; the pilot uses the same upstream
  `Dreamer`, agent, sampler, and checkpoint interfaces with bounded arguments.

| Concept | Upstream location | Role |
|---|---|---|
| RSSM | `models.py:RSSM` | Recurrent deterministic state plus stochastic latent state; `observe()` consumes posterior embeddings and `imagine()` rolls the prior forward. |
| World model | `dreamer.py:Dreamer._train`, `models.py:ConvEncoder`, `models.py:ConvDecoder`, `models.py:DenseDecoder` | Encodes pixels, predicts latent states, reconstructs images, and predicts rewards. |
| Actor | `dreamer.py:Dreamer._actor`, `models.py:ActionDecoder`, `dreamer.py:Dreamer._imagine_ahead` | Samples actions from imagined features; evaluation uses the action distribution mode. |
| Critic | `dreamer.py:Dreamer._value`, `dreamer.py:Dreamer._train` | Estimates imagined returns and receives a separate value-loss update. |
| Replay buffer | `dreamer.py:tools.save_episodes`, `tools.py:load_episodes` | Stores episode files under `logdir/episodes` and samples fixed-length sequences. |
| Imagination | `dreamer.py:Dreamer._imagine_ahead`, `models.py:RSSM.img_step` | Starts from posterior states and rolls forward through the learned transition; actor gradients pass through the frozen dynamics path. |
| Training entry point | `dreamer.py:main`, `dreamer.py:Dreamer.__call__` | Prefills episodes, trains on collection, evaluates periodically, and saves `variables.pkl`. |

These mappings were checked against the pinned source. The Kaggle pilot records the exact source commit and compatibility replacements in its manifest.
