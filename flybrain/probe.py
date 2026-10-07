r"""시각 뉴런을 좌/우로 자극하면 걷기 하행 뉴런(DN)이 어떻게 켜지나 — 오리에 잇기 전 확인.

오리 조종의 배선은 이 표에서 정한다. 좌우가 갈리지 않으면 방향을 줄 수 없고,
자극 세기에 따라 단조로 늘지 않으면 속도를 줄 수 없다.

    .venv\Scripts\python.exe flybrain\probe.py
    .venv\Scripts\python.exe flybrain\probe.py --inputs LC9 LC10a --rates 25 50 100 --ms 1000
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lif import Brain  # noqa: E402

DNS = ["DNp09", "DNa02", "DNa01", "MDN"]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=["LC9", "LC10a"])
    ap.add_argument("--rates", nargs="+", type=float, default=[50.0, 100.0])
    ap.add_argument("--ms", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    B = Brain(seed=args.seed)
    dn = {(d, s): B.find(d, s) for d in DNS for s in ("left", "right")}
    head = " | ".join(f"{d} L / R" for d in DNS)
    print(f"| 자극 | Hz | {head} | 켜진 뉴런 |")
    print("|---|---|" + "---|" * len(DNS) + "---|")
    for name in args.inputs:
        for side in ("left", "right", "both"):
            ids = B.find(name) if side == "both" else B.find(name, side)
            for hz in args.rates:
                B.reset()
                B.set_stim([(ids, hz)])
                t0 = time.time()
                c = B.run(args.ms)
                scale = 1000.0 / args.ms
                cells = []
                for d in DNS:
                    l = c[dn[(d, "left")]].mean() * scale
                    r = c[dn[(d, "right")]].mean() * scale
                    cells.append(f"{l:.0f} / {r:.0f}")
                print(f"| {name} {side} ({len(ids)}) | {hz:.0f} | " + " | ".join(cells)
                      + f" | {(c > 0).sum()} |  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
