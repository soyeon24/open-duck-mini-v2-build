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

## First closed-loop run (20 s, person at 2.0, 0.6 m)

The band stayed in view 100 % of the time with a median bearing error of 2.8°, so the steering
works. The forward drive does not: the duck covered 0.24 m. With the target near the centre the
input splits about 50 / 50, each DNp09 sits below threshold most of the time, and a 100 ms window
over a single neuron reads in 10 Hz steps. The next thing to tune is the input gain or the readout
window, not the wiring. The brain runs about 7× slower than real time (about 22× with rendering).
