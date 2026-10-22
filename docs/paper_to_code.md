# Paper to code map

The paths below correspond to the pinned `google-research/dreamer` commit. This repository does not currently copy upstream source code.

| Concept | Upstream location | Role |
|---|---|---|
| RSSM | `dreamer/models/rssm.py` | Represents latent dynamics with recurrent beliefs and stochastic states, and provides posterior/transition operations. |
| World model | `dreamer/training/define_model.py`, `dreamer/models/` | Encodes pixels, predicts latent states, reconstructs images, and predicts rewards. |
| Actor | `dreamer/control/planning.py`, `dreamer/scripts/configs.py` | Produces continuous actions from imagined latent states; sampling is used for training and mode for testing. |
| Critic | `dreamer/scripts/configs.py`, `dreamer/training/` | Estimates imagined-trajectory values to extend beyond the imagination horizon. |
| Replay buffer | `dreamer/tools/numpy_episodes.py` | Stores episode data and loads training sequences by window. |
| Imagination | `dreamer/control/planning.py`, `dreamer/training/trainer.py` | Starts from posterior states, rolls forward with learned transitions, and updates the actor/value models. |
| Training entry point | `dreamer/scripts/train.py` | Parses logdir/params and starts training. |

These mappings come from the repository file tree and configuration source. Once the source is available in the Linux environment, add exact function names and line numbers rather than inventing them before checkout.
