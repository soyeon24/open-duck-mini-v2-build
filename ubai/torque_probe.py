"""보행 중 관절 토크 실측. 부품 발주 전 서보 토크 여유 확인 (HARDWARE_PREP.md §12).

sim 액추에이터는 forcerange ±3.23 N·m 로 열려 있는데 STS3215 7.4V 실물 정격은
1.86 N·m(19 kg·cm) 다. 정책이 실물이 못 내는 토크를 쓰고 있으면 sim 걸음걸이가
실기에서 재현되지 않는다. 그걸 60만원 지르기 전에 숫자로 확인한다.

head_probe2.py 와 같은 방식 — 뷰어만 가짜로 갈아끼우고 제어 루프는 원본을 쓴다.
다만 토크 피크는 제어주기(50Hz)가 아니라 물리 스텝(500Hz)에서 나므로
mj_step 을 감싸서 매 물리 스텝마다 기록한다.
"""
import numpy as np
import mujoco
import mujoco.viewer
import playground.open_duck_mini_v2.mujoco_infer as MI

SCENE = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash_collision.xml"
REF = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"

# STS3215 7.4V 실물 정격
T_STALL = 1.86      # N·m (19 kg·cm), 스톨 토크 = 속도 0 에서만 나오는 값
W_NOLOAD = 4.71     # rad/s (0.222 s/60° = 45 rpm), 무부하 최대 속도
NM_TO_KGCM = 10.197

# 명령 = [vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll]
# 각 구간 250 제어스텝 = 5초. 앞 50 스텝(1초)은 과도구간이라 통계에서 버린다.
DWELL = 250
SETTLE = 50
# "hi"/"lo" = 그 축 명령 범위의 상한/하한. 정책이 겪어본 범위 밖으로 나가면 안 된다.
PHASES = [
    ("정지",        [0.0,  0.0,  0.0,  0, 0, 0, 0]),
    ("전진 최대",   ["hi", 0.0,  0.0,  0, 0, 0, 0]),
    ("후진 최대",   ["lo", 0.0,  0.0,  0, 0, 0, 0]),
    ("좌 게걸음",   [0.0,  "hi", 0.0,  0, 0, 0, 0]),
    ("우 게걸음",   [0.0,  "lo", 0.0,  0, 0, 0, 0]),
    ("좌회전",      [0.0,  0.0,  "hi", 0, 0, 0, 0]),
    ("우회전",      [0.0,  0.0,  "lo", 0, 0, 0, 0]),
    ("전진+좌회전", ["hi", 0.0,  "hi", 0, 0, 0, 0]),  # 조합이 보통 제일 가혹하다
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
    mj.direct_head = False  # 머리도 정책에 맡긴다. 실기에서 도는 것과 같은 조건

    state = {"ctrl_i": 0, "phase": -1}
    force, qvel, phase_of, upvec = [], [], [], []
    orig_step = mujoco.mj_step

    def rec_step(model, data, *a, **k):
        orig_step(model, data, *a, **k)
        if state["phase"] >= 0:
            force.append(data.actuator_force.copy())
            qvel.append(data.qvel[mj.actuator_qvel_addr].copy())
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

    return (np.array(force), np.abs(np.array(qvel)),
            np.array(phase_of), upvec, mj.actuator_names)


def avail(w):
    """DC 모터 토크-속도 직선 근사. 속도가 붙으면 낼 수 있는 토크가 줄어든다."""
    return np.maximum(T_STALL * (1.0 - w / W_NOLOAD), 0.0)


def report(name, f, w, ph, upvec, names):
    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)

    fell = [p for p, u in upvec if u < 0.5]
    if fell:
        bad = sorted(set(fell))
        print("⚠️ 넘어진 구간 있음 -> %s. 이 구간 수치는 못 믿는다"
              % ", ".join(PHASES[p][0] for p in bad if p < len(PHASES)))

    # 과도구간 제외 마스크 (구간별 앞 SETTLE 제어스텝 = SETTLE*decimation 물리스텝)
    keep = np.zeros(len(ph), dtype=bool)
    for p in range(len(PHASES)):
        idx = np.where(ph == p)[0]
        if len(idx):
            keep[idx[SETTLE * 10:]] = True
    a, wk = np.abs(f[keep]), w[keep]

    print("\n[관절별] 피크는 절대값. 정격 %.2f N·m (%.0f kg·cm) 기준" % (T_STALL, T_STALL * NM_TO_KGCM))
    print("%-16s %8s %8s %8s %7s %7s %7s" %
          ("관절", "피크Nm", "피크kgcm", "p99.9", ">정격%", "클램프%", "불가능%"))
    inf_all = a > avail(wk)
    for i, n in enumerate(names):
        col, wcol = a[:, i], wk[:, i]
        print("%-16s %8.2f %8.1f %8.2f %7.2f %7.2f %7.2f" % (
            n, col.max(), col.max() * NM_TO_KGCM, np.percentile(col, 99.9),
            100 * (col > T_STALL).mean(), 100 * (col > 3.22).mean(),
            100 * inf_all[:, i].mean()))

    print("\n[구간별] 다리 10축 최대 (머리 4축 제외)")
    leg = [i for i, n in enumerate(names) if not n.startswith(("neck", "head"))]
    print("%-14s %8s %8s %7s %7s   %s" % ("구간", "피크Nm", "피크kgcm", ">정격%", "불가능%", "최대 관절"))
    for p in range(len(PHASES)):
        idx = np.where(ph == p)[0][SETTLE * 10:]
        if not len(idx):
            continue
        seg, wseg = np.abs(f[np.ix_(idx, leg)]), w[np.ix_(idx, leg)]
        j = leg[int(np.unravel_index(seg.argmax(), seg.shape)[1])]
        print("%-14s %8.2f %8.1f %7.2f %7.2f   %s" % (
            PHASES[p][0], seg.max(), seg.max() * NM_TO_KGCM,
            100 * (seg > T_STALL).mean(),
            100 * (seg > avail(wseg)).mean(), names[j]))

    peak = np.abs(f[keep][:, leg]).max()
    over = 100 * (np.abs(f[keep][:, leg]) > T_STALL).mean()
    print("\n>>> 다리 피크 %.2f N·m (%.1f kg·cm) / 정격 초과 시간 %.2f%%" % (
        peak, peak * NM_TO_KGCM, over))
    print(">>> 판정: %s" % (
        "발주 진행 가능" if peak < T_STALL else
        "forcerange 를 1.86 으로 낮춰 재학습 검토" if over > 1.0 else
        "피크는 넘지만 지속 초과는 아님 — 아래 해석 참고"))


RUNS = [
    ("BEST_WALK_ONNX_2 (커뮤니티 검증 정책, 원본 명령범위)",
     "../BEST_WALK_ONNX_2.onnx", False),
    ("hw8 931375 head_pos=-8.0 300M (레퍼런스 정합 범위)",
     "../from_ubai/hw8_2026_09_06_151227_300482560.onnx", True),
]

for name, onnx, rr in RUNS:
    try:
        report(name, *probe(onnx, rr))
    except Exception as e:
        print("FAIL %s: %s: %s" % (name, type(e).__name__, e))
