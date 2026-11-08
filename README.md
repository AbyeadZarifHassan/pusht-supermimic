# SuperMimic on Push-T: pathwise (BPTT) vs likelihood-ratio (PPO) RL

This project applies the ideas from the **SuperMimic case study** (RL lectures I/II and
the SuperMimic paper) to the **Push-T** block-pushing task, using the task setup of
Ma et al. 2026, *A Single Diffusion-Policy Controller for Multi-Task Block Pushing*
(arXiv:2607.10892).

It targets the open question on the last slide of the case study:

> **Sparse rewards?** Manipulation and locomotion without a reference trajectory.

SuperMimic was shown on humanoid motion tracking, where the reward is **dense** (a
reference motion at every step). Push-T is the opposite: a contact-rich manipulation
task whose natural reward is **sparse** (1 only when the T covers 95% of the goal).

## What is implemented

| Component | File | From |
|---|---|---|
| Push-T simulator (pymunk), 95% coverage success, 300-step limit | `pusht_rl/env.py` | Push-T / Diffusion Policy |
| Objective-centric observation (block frame), velocity action in goal frame | `pusht_rl/env.py`, `pusht_rl/torch_ops.py` | ESM paper, Sec. III-B |
| Reverse curriculum: block starts within 256 ± (30 + m/M·100) px, M=10, N=50 | `pusht_rl/env.py` | ESM paper, App. B |
| Learned differentiable world model f_ω, fitted on k=8-step autoregressive windows (Eq. 1) | `pusht_rl/world_model.py` | SuperMimic Sec. 3.2 |
| Learned success/termination head, soft survival weight c_{t+1} = c_t(1 − d̂) | `pusht_rl/world_model.py` | SuperMimic Sec. 3.1 |
| Deterministic actor trained by **BPTT** through the frozen world model over H=32 steps, critic bootstrap at s_H (Eq. 2) | `pusht_rl/supermimic.py` | SuperMimic Sec. 3.3 |
| Critic fitted to λ-returns of imagined rollouts (Eq. 3) | `pusht_rl/supermimic.py` | SuperMimic Sec. 3.3 |
| Fixed-size off-policy replay buffer (old data replaced) | `pusht_rl/buffer.py` | SuperMimic |
| **PPO** baseline: clipped ratio objective + GAE | `pusht_rl/ppo.py` | RL II lecture |

### The SuperMimic loop (`train.py --algo supermimic`)

1. **Collect** – the actor plus Gaussian action noise acts in the simulator; transitions go into the replay buffer.
2. **Fit world model** – roll f_ω forward from a logged state on logged actions for k=8 steps and regress the logged states; train the success head with binary cross-entropy.
3. **Improve policy** – start from buffer states, imagine H steps with the *frozen* world model, and maximise

   J(θ) = Σ_{t<H} γ^t ĉ_t r̂_{t+1} + γ^H ĉ_H V_ψ(ŝ_H)

   by back-propagation through time. The critic V_ψ is then fitted to λ-returns of the same imagined rollout.

### Design choices for Push-T

* **World model state** (8-D): pusher position, block pose, block linear and angular velocity. The network predicts block motion **in the block frame** and rotates it back, so it does not need to relearn pushing for every position and angle.
* **Observation head** is analytic (differentiable): pusher and goal expressed in the block frame (ESM paper), plus block velocity because the Push-T block keeps sliding after a push.
* **Reward head**
  * `--reward sparse`: r̂ = P̂(success | ŝ_{t+1}) from the learned success head. The real reward is 1 on success and 0 otherwise.
  * `--reward dense`: potential-based shaping Φ(ŝ_{t+1}) − Φ(ŝ_t) + success, with Φ = exp(−d/60)·(1+cos Δθ)/2. Potential shaping keeps the optimal policy unchanged and removes any incentive to hover near the goal without finishing.
* **Success head class imbalance**: successes are rare, so each batch is 50/50 balanced and the logit is corrected by the true positive rate. This keeps the predicted termination probability calibrated.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or conda / uv
pip install -r requirements.txt                     # install a CUDA build of torch for GPU
python -m pytest tests -q                           # 6 quick checks
```

Linux, Windows or WSL all work; no MuJoCo or pygame is needed.

## Running

```bash
# main comparison (2M environment steps each)
python train.py --algo supermimic --reward dense  --run sm_dense
python train.py --algo supermimic --reward sparse --run sm_sparse
python train.py --algo ppo        --reward dense  --run ppo_dense
python train.py --algo ppo        --reward sparse --run ppo_sparse

# everything (also critic / horizon / curriculum ablations), then plot
bash run_experiments.sh
python plot.py                     # -> results.png
python evaluate.py runs/sm_dense   # success rate + GIFs of a saved checkpoint
```

Every run writes `runs/<name>/log.csv` (training), `eval.csv` (100 randomized test episodes,
same protocol as the ESM paper), `checkpoint.pt` and GIFs.

Useful flags: `--horizon H`, `--no-critic`, `--no-curriculum`, `--damping 0.99` (table friction,
see below), `--success-threshold 0.9`, `--policy-updates`, `--wm-updates`, `--n-envs`.

## Preliminary results (small budget, CPU only)

These short sanity runs were done on a 2-core CPU, far below the ~4M steps the ESM paper uses.
They check that the pipeline works; they do not compare final performance.

Training metric: mean coverage of the goal at the end of training episodes (curriculum stage 0,
block within ±30 px of the goal, random angle), averaged over the last 200 episodes, after **150k
environment steps**.

| Policy | Table friction | Final coverage | Training success |
|---|---|---|---|
| Random actions | – | 0.165 | 0% |
| **Do nothing** (zero action) | – | 0.237 | 0% |
| PPO, dense reward | none (`--damping 0`) | 0.234 | 0.5% |
| PPO, dense reward | `--damping 0.99` | 0.249 | 0% |
| SuperMimic (BPTT), dense reward | none (`--damping 0`) | 0.217 | 0% |
| **SuperMimic (BPTT), dense reward** | `--damping 0.99` | **0.374** (still rising) | **1.5%** |

What this shows:

* Without table friction, both methods end up at the **"do nothing" level**. When we inspected the
  BPTT policy, it had learned to avoid the block: with a block that keeps sliding, any push lowers
  the pose score within the 32-step horizon, so not touching it is a local optimum of J.
* With quasi-static friction, the pathwise gradient through the learned model starts to work: coverage
  rises steadily past the do-nothing level and the first real 95%-coverage successes appear.
  PPO with the same data budget has not moved yet. This matches the sample-efficiency claim
  of the case study (pathwise < likelihood ratio in samples), though only at this tiny budget.
* No run solves the task yet. The ESM paper needed ~4M steps with a diffusion policy, and PPO stayed
  at 0% there. The sparse-reward runs and the full 2M-step comparison are left for a GPU
  (`bash run_experiments.sh`). Run several seeds before drawing conclusions.
* SuperMimic is ~5× slower per environment step than PPO on CPU, because it does 4 world-model
  updates and 4 BPTT updates (256 × 32 imagined steps each) per 256 real steps. On a GPU most of
  this cost goes away.

![preliminary curves](docs/preliminary_results.png)

(`t_sm_dense_damp` = SuperMimic with friction; `t_utd4` = SuperMimic without friction; `t_ppo_*` = PPO.)

For this reason `--damping 0.99` is the **default**. Use `--damping 0` for the original Push-T physics.

## Discussion: the case-study questions on Push-T

**1. Sparse rewards.** BPTT needs ∂r/∂a. With a sparse reward this derivative is zero almost
everywhere, so all the signal has to come from (a) the learned success head, which is a smooth
surrogate of the indicator function but is only informative near states where success has been
observed, and (b) the critic. The reverse curriculum is what makes the first successes possible.
Contact makes this worse: while the pusher is not touching the block, the true dynamics give
**zero** gradient from pusher motion to block pose. Only the smoothness of the learned MLP world
model and the critic V(o), which depends on where the pusher is relative to the block, can point
the pusher toward a useful contact. The case study lists "smooth gradients through contact" as
only an assumption, and this task tests it directly.

**2. How general are the world model and policy?** The replay buffer is fixed-size, so the model
mostly knows recently visited states. The curriculum shifts the state distribution over time
(block further from the goal), and the model must keep up. Logging `wm_kstep_block_err_px` over
training shows how well it does.

**3. Does the world model help the critic?** Here the critic is trained **only** on imagined
transitions (as in SuperMimic). Compare `sm_dense` with `sm_dense_nocritic` and the horizon
sweep (`sm_dense_H8`, `H16`). SuperMimic reports that without a critic, training fails at H ≤ 16.

**4. Others.** (a) *Model exploitation*: the actor can find actions that score well in imagination
but not in the simulator. Compare `J`/`imag_success_prob` with the real `train_success`.
(b) *Local optimum "do not touch"*: with a short horizon, pushing usually lowers the pose score at
first, so BPTT can converge to standing still (see results). (c) *Physics*: the original Push-T has no table
friction, so a pushed block keeps sliding. Real tabletop pushing is closer to quasi-static, which
the default `--damping 0.99` approximates. In the runs above, this single change decided whether
BPTT learned anything, which shows how sensitive pathwise gradients are to the contact dynamics.

## Layout

```
pusht_rl/env.py          Push-T simulator, curriculum, vectorised env, renderer
pusht_rl/torch_ops.py    differentiable observation / reward / world-model I/O
pusht_rl/world_model.py  f_ω dynamics + success head, k-step loss
pusht_rl/supermimic.py   BPTT actor + λ-return critic
pusht_rl/ppo.py          PPO + GAE baseline
pusht_rl/buffer.py       per-env ring replay buffer (k-step windows)
pusht_rl/evaluation.py   100-episode evaluation + GIFs
train.py / evaluate.py / plot.py / run_experiments.sh
```
