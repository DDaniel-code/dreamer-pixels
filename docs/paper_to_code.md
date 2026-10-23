# Paper to code map

The paths below correspond to the selected `danijar/dreamer` TensorFlow 2 commit `56d4d444dfd0582b0e79dab80aebbea74c0ce40d`. The original TensorFlow 1 repository is historical reference only. This repository does not copy upstream source code.

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
