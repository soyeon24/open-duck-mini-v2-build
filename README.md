# Open Duck Mini v2 — Build Log

Building a bipedal walking duck robot from scratch.

> ### Two ducks, one folder
> The folder is named `Microduck` for historical reasons, but the actual build target is the
> **42 cm [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini)** — *not* Pollen
> Robotics' 25 cm Microduck. When the two get mixed up, read **[`ROBOTS.md`](ROBOTS.md)**:
> it is a one-screen card telling them apart, including the parts that look alike
> (Pi Zero 2 W vs Pico 2 W, STS3215 vs ST3215, 626ZZ vs 608ZZ).

Why not Microduck: its mechanical STLs and MJCF *are* published under Apache-2.0, and its
compute board is an off-the-shelf Radxa Zero 3W — but there is no BOM, no wiring diagram, no
assembly guide, and no schematic or gerber for its custom HAT PCB, so it cannot be self-built.
Full evidence in [`microduck_ref/README.md`](microduck_ref/README.md).

**Current stage: 1 — "make it walk in simulation first" (cost: $0).**
No parts ordered yet. Hardware purchase and 3D printing start once walking is confirmed in sim.

---

## Repository layout

| Path | Contents |
|---|---|
| `README.md` | This file. The map. |
| `ROBOTS.md` | **42 cm vs 25 cm — which duck is which.** Read when the two get confused. (Korean) |
| `HARDWARE_PREP.md` | Ordering, BOM, assembly pitfalls, printing, vision plan. Everything about parts. (Korean) |
| `NEXT_STEPS.md` | Plan, decision log, hardware specs, cluster access. **Read this first when resuming.** (Korean) |
| `SIM_NOTES.md` | Detailed simulation notes — standup task, head-tracking problem, findings 1–5. (Korean) |
| `ubai/` | SLURM scripts for the UBAI supercomputer, plus the local measurement probes (`head_probe2.py`, `torque_probe.py`, `torque_derate_test.py`). Procedure in `ubai/README.md`. (Korean) |
| `ubai_standup/` | sbatch script for standup training |
| `from_ubai/` | 9 trained ONNX policies retrieved from the cluster |
| `print/` | 36 STL types (51 parts) for 3D printing + `PRINT_CHECKLIST.md` |
| `microduck_ref/` | Research record on Pollen's Microduck (25 cm) — **the robot this project does not build.** Kept as the reasoning behind the choice. (Korean) |
| `make_standup_xml.py` | Generates the standup training XML (adds ground collision boxes) |
| `smoke_standup.py` | Local CPU sanity check for the standup env (not training — a pre-flight check) |

Two paths referenced below are **not** in this repository:

- **`Open_Duck_Playground/`** — the RL training environment. It lives in a separate
  [fork of `apirrone/Open_Duck_Playground`](https://github.com/soyeon24/Open_Duck_Playground)
  (branch `standup-task-and-fixes`); see *Attribution & License* below. Clone it next to this repo.
- **`BEST_WALK_ONNX_2.onnx`** — the community-validated walking policy used as a baseline.
  Not redistributed here (see below). Obtain it from
  [Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) or the project Discord.

---

## Split workflow — campus vs. home

There is no local GPU (Intel Iris Xe), so training runs on the **UBAI SLURM cluster at the
University of Seoul**. The UBAI gateway sits on a private IP and there is no VPN, so it is
**reachable only from the campus network**.

| Where | What |
|---|---|
| **Campus** | Submit training with `sbatch`, retrieve result ONNX files via `scp` |
| **Home** | Prepare code and scripts, replay and analyze retrieved policies on local CPU |

Hence the rhythm: **queue up a batch on each campus visit, collect it on the next one.**
An `ssh` timeout from home is expected behavior, not a fault.

> Acknowledgement required for any publication using these resources:
> "The authors acknowledge the Urban Big data and AI Institute of the University of Seoul
> supercomputing resources (http://ubai.uos.ac.kr) made available for conducting the research
> reported in this paper."

---

## Quick start — watch a policy run

```
cd <repo_parent>\Open_Duck_Playground
set PYTHONPATH=<repo_parent>\Open_Duck_Playground
..\.venv\Scripts\python.exe -u -m playground.open_duck_mini_v2.mujoco_infer -o ..\BEST_WALK_ONNX_2.onnx
```

For standup policies, point at the scene that has torso collision geometry:
`--model_path playground\open_duck_mini_v2\xmls\scene_standup.xml`

### Controls (code comments assume AZERTY; these are the actual keys)

| Key | Action |
|---|---|
| ↑ ↓ | Forward / backward |
| ← → | Strafe left / right |
| Q / E | Turn left / right |
| P / ; | Gait frequency ±0.1 |
| **R** | **Full reset** (added by this project. Do not use Backspace — it falls over instantly) |
| 1–5 / 0 | Five dance moves / stop |
| T | Toggle direct head control |

Commands latch until changed. The viewer window must have focus.

---

## Two Python environments — do not mix them

| | Purpose | Versions |
|---|---|---|
| `.venv` | **Inference and viewer only** | jax 0.11.1 / mujoco 3.12.0 / playground 0.2.0 |
| `.venv-train-cpu` | **Local validation of training code** (matches the cluster) | jax 0.6.2 / mujoco 3.3.4 / playground 0.0.5 |

`joystick.py` and `standup.py` cannot be imported under `.venv`
(`mujoco_playground._src.collision` was removed in playground 0.1.0).
Inference only needs onnxruntime, so it runs fine on the newer stack.

---

## Done so far

- [x] Simulation environment set up; walking confirmed with the community policy
- [x] UBAI cluster access, environment install, GPU training pipeline (`ubai/`)
- [x] **Completed a 300M-step walking run** — 1 h 11 min, 68,700 steps/s (RTX 3090)
- [x] **Designed a standup task from scratch** — not present upstream. Iterated v1 → v3
- [x] Viewer improvements — R-key reset, CPU usage 98% → 3.6%, five dance moves
- [x] Found 5 upstream bugs (collision geometry, ignored head commands, checkpoint resume, sensor addressing)
- [x] Collected 51 STL parts + print plan (PLA 990 g + TPU 34 g, Bambu H2D/X1C)
- [x] **Measured actuator torque headroom before committing to the parts order** — the sim
      allows ±3.23 N·m per joint but a real STS3215 stalls at 1.86 N·m. The knee sits on that
      clamp 14–16 % of the time, yet derating the model to the real limit never made either
      policy fall; it only cost speed (−42 % / −69 %). Parts order unblocked

## Next

- [ ] Compare the three walking + head-tracking policies in the viewer — all three jobs
      (910949 / 910953 / 910954) completed and were retrieved into `from_ubai/` on 2026-09-06.
      Note the final 300M checkpoint scored *lower* than the 279M one in all three runs, so both
      were kept; the gap is within one reward std, so the viewer has to settle it
- [ ] Verify whether full inversion recovery is **physically possible at all** (no arms — it may not be)
- [ ] Retrain one 300 M run with `forcerange` set to the real servo limit (1.86 N·m) so the
      policy stops relying on torque the hardware cannot produce (~1 h 22 min on an A6000)
- [ ] Fix `--restore_checkpoint_path` — needed for runs longer than 48 h
- [ ] After sim validation → order parts + 3D print → assemble → **sim2real tuning (the real wall, 2–6 weeks)**

Estimated total: 2–4 months, roughly 550,000–650,000 KRW in parts.

---

## Changes made to upstream `Open_Duck_Playground`

These live in the fork, not in this repository. Originals are kept as `.orig` backups locally.

| File | Change |
|---|---|
| `mujoco_infer.py` | R-key full reset, `viewer.sync` cadence, dance moves, direct head control |
| `onnx_infer.py` | onnxruntime single-threaded, spin-wait disabled (fixes CPU pegging) |
| `mujoco_infer_base.py` | Fix `get_gravity` sensor address bug |
| `rewards.py` | Add `only_when_moving` flag to `cost_head_pos` |
| `joystick.py` | Register the `head_pos` reward, align command ranges with the reference motion |
| `standup.py` (new) | Standup task. Subclasses `Joystick`, so the observation layout matches and existing viewers can replay it |
| `standup_runner.py` (new) | Standup training entry point (kept separate so `runner.py` stays untouched) |
| `*_standup.xml` (new) | Model + scene with ground collision boxes |

Rationale and dead ends are documented in `SIM_NOTES.md`.

---

## Attribution & License

This repository contains **two kinds of material**, licensed differently.

### 1. Original work — this project

`ubai/`, `ubai_standup/`, `make_standup_xml.py`, `smoke_standup.py`,
`NEXT_STEPS.md`, `SIM_NOTES.md`, `print/PRINT_CHECKLIST.md`, this README, and the
trained policies in `from_ubai/` are the author's own work.

### 2. Third-party material

| Material | Origin | License |
|---|---|---|
| `print/*.stl`, `print/print_guide_original.md` | [apirrone/Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini) (v2 branch) | **Apache-2.0** — see [`print/LICENSE`](print/LICENSE) and [`print/NOTICE.md`](print/NOTICE.md) |
| Training code (in the separate fork) | [apirrone/Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) | See note below |

**Note on the training code.** `apirrone/Open_Duck_Playground` publishes **no LICENSE file**.
Individual files derived from [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground)
(`joystick.py`, `standing.py`, `base.py`, `constants.py`, `randomize.py`) do carry
Apache-2.0 headers, and modifications to those files are marked in place as Apache-2.0 §4(b)
requires. The remaining files carry no license grant. For that reason the modified training
code is published **as a GitHub fork of the upstream repository** rather than copied into this
repository — forking is explicitly permitted by the GitHub Terms of Service (D.5) — and
`BEST_WALK_ONNX_2.onnx` is **not redistributed** here.

If you are the upstream author and would like anything here changed or removed, please open an issue.

---

## Links

- Main repo (hardware / CAD): https://github.com/apirrone/Open_Duck_Mini — **v2 branch**
- Training: https://github.com/apirrone/Open_Duck_Playground
- On-robot runtime: https://github.com/apirrone/Open_Duck_Mini_Runtime
- Reference motion generation: https://github.com/apirrone/Open_Duck_reference_motion_generator
- Korean overview: https://robotics.growbotics.ai/ko/projects/hardware/open-duck-mini-v2
- Discord: https://discord.gg/UtJZsgfQGe
