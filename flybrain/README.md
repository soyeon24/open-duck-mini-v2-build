# A fruit-fly brain steering the duck

The whole-brain leaky integrate-and-fire model of *Drosophila* from
[Shiu et al. 2024, *Nature*](https://github.com/philshiu/Drosophila_brain_model), run on the
FlyWire 783 connectome (138,639 neurons, 15.1 M connections), drives the duck's walking policy.

```
head camera → ankle-band bearing → left / right LC9 Poisson input
  → whole-brain LIF, 20 ms per control step
  → descending-neuron rates → (dx, dyaw) command for the walking policy
```

**Why LC9.** The forward-walking descending neuron P9 (DNp09) gets strong input from the LC9
visual projection neurons and is required for a male to pursue a female
([Bidaye et al. 2020](https://www.biorxiv.org/content/10.1101/798439.full.pdf)). Following a person
is handed to the fly's own pursuit pathway.

## Files

| File | What it does |
|---|---|
| `build_brain.py` | One-off: turns the downloaded tables into `data/brain_783.npz` (needs pandas + pyarrow) |
| `lif.py` | The Shiu et al. model in numpy, same equations and constants as the Brian2 original. Event-driven spike delivery, exact integration of the linear part, dt = 0.1 ms |
| `probe.py` | Stimulates visual neurons on one side and tabulates descending-neuron rates. The wiring in `fly_pilot.py` comes from this table |
| `../fly_pilot.py` | Closed loop in MuJoCo with the default walking policy |

## Data (not committed, ~136 MB)

```
flybrain/data/Completeness_783.csv       https://github.com/philshiu/Drosophila_brain_model
flybrain/data/Connectivity_783.parquet   same repository
flybrain/data/neuron_annotations.tsv     https://github.com/flyconnectome/flywire_annotations
                                         (supplemental_files/Supplemental_file1_neuron_annotations.tsv)
```

Then `python flybrain/build_brain.py` once.

## What the connectome does (probe.py, 1 s per row, Hz)

| Input | Rate | DNp09 L / R | DNa02 L / R | DNa01 L / R | MDN L / R |
|---|---|---|---|---|---|
| LC9 left (87) | 50 | 20 / 0 | 5 / 0 | 13 / 41 | 1 / 0 |
| LC9 left | 100 | 41 / 0 | 9 / 0 | 7 / 58 | 4 / 2 |
| LC9 right (92) | 50 | 0 / 35 | 3 / 43 | 74 / 0 | 10 / 8 |
| LC9 right | 100 | 0 / 73 | 9 / 71 | 98 / 0 | 27 / 21 |
| LC9 both (179) | 100 | 30 / 2 | 0 / 0 | 76 / 29 | 6 / 2 |
| LC10a left (115) | 100 | 0 / 0 | 107 / 0 | 8 / 0 | 0 / 0 |
| LC10a right (119) | 100 | 0 / 0 | 0 / 5 | 3 / 1 | 0 / 0 |

- LC9 on one side drives DNp09 on **the same side only**, and more input gives more output.
  That is the forward drive with ipsilateral turning that P9 is known for.
- DNa01 comes on **contralateral** to the input, which would steer the wrong way, so it is not
  read out.
- The brain is not symmetric. Driving both LC9 populations equally favours the left DNp09, and
  LC10a works on the left only. LC10a is not used.

Readout: `dx = KX·(DNp09 L + R) − KB·MDN`, `dyaw = KY·((DNp09 L − R) + (DNa02 L − R))`.

## Left and right inhibit each other

A 1 s grid of left / right LC9 rates (Hz) against DNp09 left / right (Hz):

| LC9 L / R | 150 / 0 | 100 / 100 | 50 / 50 | 150 / 150 |
|---|---|---|---|---|
| DNp09 L / R | 74 / 0 | 22 / 4 | 5 / 1 | 46 / 9 |

Equal input on both sides silences the forward neurons. The first wiring split the input 50 / 50
for a target straight ahead (8° split width), so it sat in exactly that dead zone: 0.24 m in 20 s.
The current wiring makes one side win (3° split width, 200 Hz), and the duck pursues the way a fly
does, with small left and right corrections. Rates are now smoothed with a 150 ms exponential
filter instead of a 100 ms box over a single neuron, which read in 10 Hz steps.

## Closed loop (30 s, person at 2.0, 0.6 m, start facing +x)

| Wiring | Distance 2.09 m → | Time to 0.70 m | Band in view |
|---|---|---|---|
| 8° split, 100 Hz (first) | 1.85 m after 20 s | — | 100 % |
| 3° split, 150 Hz | 0.70 m | 14 s | 100 % |
| **3° split, 200 Hz (default)** | **0.68 m** | **10 s** | 100 % |

There is no stop rule in the readout. Once the person is close, the input drops (`--near`/`--far`), the two
sides are again roughly equal, they suppress each other, and the duck stands about 0.7 m in front.

The brain runs about 7× slower than real time (about 40× with the head-camera render), so
`fly_pilot.py` simulates headless, saves the trajectory, and replays it in the MuJoCo viewer in real
time with LC9 input and DN rates overlaid.

## A walking person: fly brain vs. the existing follower

Same five walking-person scripts, same verdicts (`eval_follow_moving.summarize`), same walking
policy (hp0dy2), empty floor, person at 0.1 and 0.2 m/s. The existing follower is
`MjInfer.follow_step` (`eval_follow_moving.py --speeds 0.1 0.2`); the fly brain is
`eval_fly_moving.py` (`--no_head` for the first fly column). Cells are the 0.1 / 0.2 m/s verdicts.

| Script | Existing follower | Fly: head fixed | + head turns | + ground distance | **+ input ramp 0.45–0.7 m (default)** |
|---|---|---|---|---|---|
| Walk away | OK / OK | OK / OK | OK / OK | OK / OK | **OK / OK** |
| Cross in front | OK / OK | lost / lost | lost / OK | OK / OK | **OK / OK** |
| Walk away, turn left | OK / OK | lost / lost | lost / OK | OK / OK | **OK / OK** |
| Walk up to 25 cm | OK / OK | lost / lost | lost / lost | lost / lost | lost / lost |
| Half circle, r 1 m | OK / OK | lost / lost | too close / OK | lost / lost | lost / lost |
| **Total** | **10 / 10** | **2 / 10** | **5 / 10** | **6 / 10** | **6 / 10** |
| Stop distance, passing runs | 0.63–0.66 m | 0.64–0.67 m | 0.53–0.70 m | 0.69–0.81 m | **0.62–0.71 m** |

What each step changed:

1. **Head fixed (2/10).** The brain keeps up exactly while the person stays inside the camera's
   ±31° view. Every failure starts with the band leaving the image sideways; with no input the
   brain goes quiet and the duck stands still.
2. **Head turns (5/10).** The head is turned outside the brain to keep the person centred, within
   ±25° (the existing follower's limit), and the brain receives the bearing in the body frame
   (image bearing + head yaw). Crossing and turning now work at 0.2 m/s but not at 0.1. At 0.1 the
   duck caught up and kept going to 0.39 m: the distance came from the band's width, which doubles
   when the two ankles overlap seen from the side, so the "reduce input when close" rule never
   fired, and the band dropped below the camera.
3. **Ground distance (6/10).** The distance now comes from projecting the band onto the floor,
   as the existing follower does. Crossing and turning pass at both speeds without losing the
   person for a single frame. The half circle passed in column 3 only because the inflated
   distance kept the input high.
4. **Input ramp moved in, 0.9 → 0.7 m (6/10).** The stop distance now matches the existing
   follower. The half circle still fails, and the reason turned out not to be the input. Near
   the person the readout asks for dx ≈ 0.12 and dyaw ≈ 0.2–0.4 rad/s, and the duck turns at
   0–5°/s. Both commands sit in the walking policy's dead zone, which is already measured in
   `mujoco_infer.py`: turning on the spot needs a yaw command above 0.5, and forward commands
   below 0.1 barely move. The person circles at 11.5°/s, the head pins at 25°, and the band leaves
   the view at 56°. The existing follower jumps the dead zone (it turns on the spot at 0.8).

Still missing, against the existing follower: turning through the policy's dead zone, and backing
off when someone walks up (no MDN drive is wired to closeness).
