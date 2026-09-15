"""보행 중 팩 전류 실측. BMS·셀 정격을 고르기 위한 것 (BOM.md §2.3 의 "10A 추정" 을 대체).

torque_probe.py 가 잰 것은 **관절 토크**였다. 발주에 필요한 건 그게 아니라 **팩 전류**다.
DC 모터는 토크가 전류에 비례하므로 (tau = Kt * I) 14축 토크를 전부 더해서 환산하면 된다.

    I(tau) = I_noload + |tau| * (I_stall - I_noload) / T_stall

STS3215 데이터시트는 6V 기준이라 7.4V 로 스케일한다 (스톨에서 I = V/R 이므로 전압 비례).

왜 이 숫자가 중요한가 — 두 가지가 걸려 있다:
  1. BMS 과전류 컷: 모자라면 보행 중에 끊긴다
  2. **전압 강하**: 팩 내부저항 때문에 V_load = V_open - I*R 로 처지고,
     STS3215 는 토크가 전압에 비례하므로 그대로 토크 손실이 된다.
     README 실측: 6.8V(처진 팩) -> 1.71 N.m -> fr186 속도 -15%

torque_probe.py 와 같은 방식 — 뷰어만 가짜로 갈아끼우고 제어 루프는 원본을 쓴다.
"""
import io
import os
import sys
import pathlib

# Windows 콘솔 기본 cp949 라 한글이 깨진다.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 어디서 실행하든 동작하도록 Open_Duck_Playground 로 옮겨간다 (SCENE/REF 이 상대경로다).
_PG = pathlib.Path(__file__).resolve().parent.parent / "Open_Duck_Playground"
os.chdir(_PG)
sys.path.insert(0, str(_PG))

import numpy as np
import mujoco
import mujoco.viewer
import playground.open_duck_mini_v2.mujoco_infer as MI

SCENE = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash_collision.xml"
REF = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"

# --- STS3215 7.4V 전기 사양 ---
# 데이터시트 표기는 6V 기준: 스톨 2.0A, 무부하 0.15A, 피크토크 19.5 kg.cm
# 스톨에서 I = V/R 이므로 전압에 비례해 스케일한다.
V_NOM = 7.4
V_SPEC = 6.0
I_STALL = 2.0 * (V_NOM / V_SPEC)     # ~2.47 A
I_NOLOAD = 0.15 * (V_NOM / V_SPEC)   # ~0.185 A
T_STALL = 1.86                       # N.m, 7.4V 정격 (HARDWARE_PREP.md §12)
KT_INV = (I_STALL - I_NOLOAD) / T_STALL   # A per N.m

# --- 팩 내부저항 (전압 강하 계산용) ---
# 30Q 셀 1개 ~25 mohm x 2 직렬 + BMS MOSFET ~15 mohm + 배선/커넥터 ~15 mohm
R_PACK = 0.025 * 2 + 0.015 + 0.015   # ohm

DWELL = 250
SETTLE = 50
PHASES = [
    ("정지",        [0.0,  0.0,  0.0,  0, 0, 0, 0]),
    ("전진 최대",   ["hi", 0.0,  0.0,  0, 0, 0, 0]),
    ("후진 최대",   ["lo", 0.0,  0.0,  0, 0, 0, 0]),
    ("좌 게걸음",   [0.0,  "hi", 0.0,  0, 0, 0, 0]),
    ("우 게걸음",   [0.0,  "lo", 0.0,  0, 0, 0, 0]),
    ("좌회전",      [0.0,  0.0,  "hi", 0, 0, 0, 0]),
    ("우회전",      [0.0,  0.0,  "lo", 0, 0, 0, 0]),
    ("전진+좌회전", ["hi", 0.0,  "hi", 0, 0, 0, 0]),
]


class Done(Exception):
    pass


def resolve(cmd, mj):
    rng = [mj.COMMANDS_RANGE_X, mj.COMMANDS_RANGE_Y, mj.COMMANDS_RANGE_THETA]
    out = list(cmd)
    for i in range(3):
        if out[i] == "hi":
            out[i] = rng[i][1]
        elif out[i] == "lo":
            out[i] = rng[i][0]
    return [float(v) for v in out]


def probe(onnx, ref_range):
    mj = MI.MjInfer(SCENE, REF, onnx, False, ref_range)
    mj.direct_head = False

    state = {"ctrl_i": 0, "phase": -1}
    force, phase_of, upvec = [], [], []
    orig_step = mujoco.mj_step

    def rec_step(model, data, *a, **k):
        orig_step(model, data, *a, **k)
        if state["phase"] >= 0:
            force.append(data.actuator_force.copy())
            phase_of.append(state["phase"])

    def sync():
        i = state["ctrl_i"]
        state["ctrl_i"] += 1
        p, off = divmod(i, DWELL)
        if p >= len(PHASES):
            raise Done
        if off == 0:
            mj.commands = resolve(PHASES[p][1], mj)
        state["phase"] = p
        upvec.append((p, mj.get_gravity(mj.data)[-1]))

    fake = type("V", (), {"sync": staticmethod(sync),
                          "__enter__": lambda s: s,
                          "__exit__": lambda *a: False})()
    mujoco.viewer.launch_passive = lambda *a, **k: fake
    mujoco.mj_step = rec_step
    MI.time.sleep = lambda *a: None
    mj.full_reset()
    try:
        mj.run()
    except Done:
        pass
    finally:
        mujoco.mj_step = orig_step

    return np.array(force), np.array(phase_of), upvec, mj.actuator_names


def pack_current(f):
    """14축 토크 -> 팩 전류(A). 실물 서보는 정격 토크를 못 넘으므로 거기서 자른다."""
    tau = np.minimum(np.abs(f), T_STALL)
    return (I_NOLOAD + tau * KT_INV).sum(axis=1)


def report(name, f, ph, upvec, names):
    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)

    fell = [p for p, u in upvec if u < 0.5]
    if fell:
        bad = sorted(set(p for p in fell if p < len(PHASES)))
        print("!! 넘어진 구간 -> %s. 이 구간 수치는 못 믿는다"
              % ", ".join(PHASES[p][0] for p in bad))

    keep = np.zeros(len(ph), dtype=bool)
    for p in range(len(PHASES)):
        idx = np.where(ph == p)[0]
        if len(idx):
            keep[idx[SETTLE * 10:]] = True

    print("\n[모터 모델] 스톨 %.2fA / 무부하 %.2fA @ %.1fV, 정격토크 %.2f N.m"
          % (I_STALL, I_NOLOAD, V_NOM, T_STALL))
    print("[팩 내부저항] %.0f mohm (셀2 + BMS + 배선)" % (R_PACK * 1000))

    print("\n[구간별] 팩 전류 = 14축 합")
    print("%-14s %7s %7s %7s %8s %8s" %
          ("구간", "평균A", "p99.9", "피크A", "피크강하V", "그때전압"))
    for p in range(len(PHASES)):
        idx = np.where(ph == p)[0][SETTLE * 10:]
        if not len(idx):
            continue
        i_pack = pack_current(f[idx])
        drop = i_pack.max() * R_PACK
        print("%-14s %7.2f %7.2f %7.2f %8.2f %8.2f" % (
            PHASES[p][0], i_pack.mean(), np.percentile(i_pack, 99.9),
            i_pack.max(), drop, V_NOM - drop))

    ip = pack_current(f[keep])
    print("\n>>> 전체: 평균 %.2f A / p99.9 %.2f A / 피크 %.2f A" %
          (ip.mean(), np.percentile(ip, 99.9), ip.max()))
    for thr in (5, 8, 10, 15):
        print("    %2d A 초과 시간 %6.2f%%" % (thr, 100 * (ip > thr).mean()))

    drop = ip.max() * R_PACK
    print(">>> 피크에서 전압 강하 %.2f V -> %.2f V (토크 -%.0f%%)"
          % (drop, V_NOM - drop, 100 * drop / V_NOM))

    # 3000mAh 기준 지속시간 (평균 전류로)
    print(">>> 3000mAh 기준 연속보행 추정 %.0f 분 (평균 %.2f A)"
          % (60 * 3.0 / ip.mean(), ip.mean()))


RUNS = [
    ("fr186 300M (실기용 정책, 1.86 으로 학습)",
     "../from_ubai/fr186_2026_09_12_174638_300482560.onnx", True),
    ("BEST_WALK_ONNX_2 (커뮤니티 검증 정책)",
     "../BEST_WALK_ONNX_2.onnx", False),
]

for name, onnx, rr in RUNS:
    try:
        report(name, *probe(onnx, rr))
    except Exception as e:
        print("FAIL %s: %s: %s" % (name, type(e).__name__, e))
