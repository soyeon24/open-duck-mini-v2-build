"""머리 추종/자발적 움직임 실측. 뷰어만 가짜로 바꾸고 제어 루프는 원본을 그대로 쓴다."""
import numpy as np
import mujoco
import mujoco.viewer
import playground.open_duck_mini_v2.mujoco_infer as MI

HEAD = {"neck_pitch": 5, "head_pitch": 6, "head_yaw": 7, "head_roll": 8}
CMD_YAW = 5  # commands[5] = head_yaw
SCENE = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash_collision.xml"
REF = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"

PHASES = [
    (0, [0.0, 0, 0, 0, 0, 0.0, 0]),    # 제자리, 머리 명령 없음
    (250, [0.15, 0, 0, 0, 0, 0.0, 0]),  # 전진, 머리 명령 없음
    (500, [0.0, 0, 0, 0, 0, 1.5, 0]),   # 제자리, head_yaw +1.5
    (750, [0.0, 0, 0, 0, 0, -1.5, 0]),  # 제자리, head_yaw -1.5
]


class Done(Exception):
    pass


def probe(onnx, ref_range):
    mj = MI.MjInfer(SCENE, REF, onnx, False, ref_range)
    mj.direct_head = False  # 머리를 정책에 맡긴다 (T 키 OFF 상태)
    rec, state = [], {"n": 0}

    def sync():
        i = state["n"]
        state["n"] += 1
        for start, cmd in PHASES:
            if i == start:
                mj.commands = list(cmd)
        rec.append((i, mj.get_actuator_joints_qpos(mj.data.qpos).copy()))
        if i >= PHASES[-1][0] + 200:
            raise Done

    fake = type("V", (), {"sync": staticmethod(sync),
                          "__enter__": lambda s: s,
                          "__exit__": lambda *a: False})()
    mujoco.viewer.launch_passive = lambda *a, **k: fake
    MI.time.sleep = lambda *a: None
    mj.full_reset()
    try:
        mj.run()
    except Done:
        pass
    return rec


def stats(rec, k):
    """구간 k 의 관절별 (평균, 진폭). 과도구간 50스텝은 버린다."""
    start = PHASES[k][0]
    end = PHASES[k + 1][0] if k + 1 < len(PHASES) else rec[-1][0] + 1
    q = np.array([r[1] for r in rec if start + 50 <= r[0] < end])
    return {n: (q[:, i].mean(), q[:, i].max() - q[:, i].min()) for n, i in HEAD.items()}


RUNS = [
    ("fast 910953  head_pos=-1.0 (대조군)", "../from_ubai/fast_2026_09_04_141412_300482560.onnx"),
    ("hw3  931374  head_pos=-3.0", "../from_ubai/hw3_2026_09_06_151149_300482560.onnx"),
    ("hw8  931375  head_pos=-8.0", "../from_ubai/hw8_2026_09_06_151227_300482560.onnx"),
]

res = {}
for name, onnx in RUNS:
    try:
        rec = probe(onnx, True)
        res[name] = [stats(rec, k) for k in range(len(PHASES))]
    except Exception as e:
        print("FAIL " + name + ": " + type(e).__name__ + ": " + str(e))

R2D = 57.29578
print("\n=== head_yaw 명령 추종 (rad) - 명령 +-1.5, 관절 가동범위 +-2.793 ===")
print("%-38s %8s %8s %8s %8s" % ("정책", "cmd 0", "cmd+1.5", "cmd-1.5", "스윙"))
for name in res:
    a = res[name][0]["head_yaw"][0]
    b = res[name][2]["head_yaw"][0]
    c = res[name][3]["head_yaw"][0]
    print("%-38s %+8.3f %+8.3f %+8.3f %8.3f" % (name, a, b, c, abs(b - c)))

print("\n=== 명령 0 일 때 정지 자세 오차 (도, 정답 0) ===")
print("%-38s %10s %10s %10s %10s" % ("정책", "neck_pitch", "head_pitch", "head_yaw", "head_roll"))
for name in res:
    s = res[name][0]
    print("%-38s %+10.1f %+10.1f %+10.1f %+10.1f" % (
        name, s["neck_pitch"][0] * R2D, s["head_pitch"][0] * R2D,
        s["head_yaw"][0] * R2D, s["head_roll"][0] * R2D))

print("\n=== 전진 중 머리 진폭 (rad) - 걸을 때 흔들림 ===")
print("%-38s %12s %12s" % ("정책", "head_pitch", "head_yaw"))
for name in res:
    s = res[name][1]
    print("%-38s %12.3f %12.3f" % (name, s["head_pitch"][1], s["head_yaw"][1]))
