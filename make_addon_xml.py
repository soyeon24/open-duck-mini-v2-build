r"""머리에 물건 올리기(carry) · 스케이트(skate) 학습용 씬 생성.

원본 open_duck_mini_v2_backlash.xml / scene_flat_terrain_backlash.xml 은 건드리지 않는다
(서버의 걷기 잡이 그걸 쓴다). make_standup_xml.py 와 같은 방식으로 복사본을 만든다.

carry — 머리 꼭대기에 납작한 쟁반(12 x 12 cm, 3 mm)을 붙이고 그 위에 물건을 올린다.
    머리 메시 윗면이 평평하지 않아서, 실물이라면 3D 프린트 판을 붙일 자리다.
    쟁반은 body `carry_tray` 의 원점이 **쟁반 윗면 중앙**이고 z 축이 위를 보게 둔다
    (학습 환경이 이 프레임 기준으로 물건 위치를 잰다).
    물건은 자유 물체 `carry_object` — 상자(5 cm 정육면체) 또는 공(지름 5 cm).
    충돌: 쟁반 contype 2 / 물건 conaffinity 3 → 물건은 쟁반·바닥과만 부딪는다.
    쟁반은 바닥과 안 부딪는다. 물건-발은 <exclude> 로 뺀다 (MJX 계산량).

skate — 양발 밑창 아래에 인라인 바퀴 3개씩. 바퀴는 구(반지름 16 mm)에 발 좌우축 힌지,
    구동 없음(수동). 구라도 힌지 축으로만 굴러서 옆으로는 마찰로 버틴다 = 인라인 바퀴.
    프레임(바퀴 축받이)은 밑창 아래 4 mm. 그래서 몸이 2r+4mm = 36 mm 높아진다.
    원래 밑창 충돌은 남긴다 — 발끝·뒤꿈치를 크게 기울이면 바닥에 닿는 브레이크가 된다.
    바퀴 관절은 frictionloss 0 (randomize.py 가 frictionloss 있는 dof 를 모터로 센다).

keyframe 'home' 은 원본 키프레임을 관절 **이름**으로 옮겨 다시 만든다. 늘어난 관절
(물건 자유관절, 바퀴 6개)은 0 / 쟁반 위 자리로 채운다.

사용:  .venv\Scripts\python.exe make_addon_xml.py [carry] [skate]   (안 주면 둘 다)
"""
import copy
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
XMLDIR = ROOT / "Open_Duck_Playground/playground/open_duck_mini_v2/xmls"
ROBOT_SRC = XMLDIR / "open_duck_mini_v2_backlash.xml"
SCENE_SRC = XMLDIR / "scene_flat_terrain_backlash.xml"

sys.path.insert(0, str(ROOT / "Open_Duck_Playground"))
from playground.open_duck_mini_v2 import base  # noqa: E402

# ── carry ────────────────────────────────────────────────────────────────
TRAY_HALF = 0.06          # 쟁반 반폭 (m)
TRAY_HALF_T = 0.0015      # 쟁반 반두께
TRAY_MASS = 0.02
OBJ_HALF = 0.025          # 상자 반변 / 공 반지름
OBJ_MASS = 0.05           # 명목값. 학습에서는 carry.py 가 U(0.02, 0.15) 로 바꾼다
OBJ_FRICTION = 0.6        # 명목값. 학습에서는 U(0.4, 1.0)
# 머리 body 프레임에서 'head' 사이트 (머리 윗면 중앙) 위치. 머리 로컬 +x 가 월드 위다.
HEAD_TOP_LOCAL = np.array([0.04245, 0.0, 0.03595])

# ── skate ────────────────────────────────────────────────────────────────
WHEEL_R = 0.016
WHEEL_GAP = 0.004         # 밑창과 바퀴 사이 (프레임 두께)
WHEEL_MASS = 0.010
FRAME_MASS = 0.020        # 발마다 프레임
WHEEL_DAMPING = 0.0005    # 베어링 저항 (N·m·s/rad)
N_WHEELS = 3


def load(xml_text):
    return mujoco.MjModel.from_xml_string(xml_text, assets=assets())


def assets():
    a = base.get_assets()
    # 새로 쓴 로봇 xml 을 include 가 찾을 수 있게 매번 다시 읽는다.
    for p in XMLDIR.glob("*.xml"):
        a[p.name] = p.read_bytes()
    return a


def bodies_by_name(root):
    out = {}
    for b in root.iter("body"):
        out[b.get("name")] = b
    return out


def sole_frame(m, d, body):
    """발 body 로컬 프레임에서 밑창(foot_bottom_tpu) 메시의 범위."""
    bid = m.body(body).id
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) != "foot_bottom_tpu":
            continue
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        P = v @ R.reshape(3, 3).T + m.geom_pos[g]
        return P.min(0), P.max(0)
    raise RuntimeError(f"{body}: foot_bottom_tpu 메시가 없다")


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def write_scene(robot_file, scene_dst, extra_world=None, extra_contact=None, qpos_fill=None):
    """원본 씬을 복사해 include 를 바꾸고, 키프레임을 관절 이름으로 다시 만든다."""
    text = SCENE_SRC.read_text(encoding="utf-8")
    text = text.replace('<include file="open_duck_mini_v2_backlash.xml"/>',
                        f'<include file="{robot_file}"/>')
    key_re = re.compile(r"<keyframe>.*?</keyframe>", re.S)
    old_key = key_re.search(text).group(0)
    text_nokey = key_re.sub("", text)
    if extra_world:
        # 마지막 </worldbody> 에 넣는다. 원본 씬 위쪽에 주석 처리된 <worldbody> 가 있다.
        i = text_nokey.rfind("</worldbody>")
        text_nokey = text_nokey[:i] + extra_world + "\n    " + text_nokey[i:]
    if extra_contact:
        text_nokey = text_nokey.replace("</mujoco>", extra_contact + "\n</mujoco>")

    m_old = load(text.replace(f'<include file="{robot_file}"/>',
                              '<include file="open_duck_mini_v2_backlash.xml"/>'))
    key_old = m_old.keyframe("home")
    m_new = load(text_nokey)

    qpos = m_new.qpos0.copy()
    for j in range(m_old.njnt):
        name = m_old.jnt(j).name
        a_old = m_old.jnt_qposadr[j]
        a_new = m_new.joint(name).qposadr[0]
        n = {0: 7, 1: 4, 2: 1, 3: 1}[int(m_old.jnt_type[j])]
        qpos[a_new:a_new + n] = key_old.qpos[a_old:a_old + n]
    if qpos_fill is not None:
        qpos = qpos_fill(m_new, qpos)
    ctrl = key_old.ctrl.copy()

    key = ("<keyframe>\n        <key name=\"home\"\n            qpos=\"" + fmt(qpos)
           + "\"\n            ctrl=\"" + fmt(ctrl) + "\"/>\n    </keyframe>")
    text_new = text_nokey.replace("</mujoco>", "    " + key + "\n</mujoco>")
    scene_dst.write_text(text_new, encoding="utf-8")
    m = load(text_new)
    print(f"  {scene_dst.name}: nq {m.nq} nv {m.nv} nu {m.nu} ngeom {m.ngeom}")
    return m


# ═════════════════════════════════════════════════════════════════════════
def make_carry():
    tree = ET.parse(ROBOT_SRC)
    root = tree.getroot()
    head = bodies_by_name(root)["head_assembly"]
    # 쟁반 프레임: 머리 로컬 y 축을 중심으로 +90° → 쟁반 z = 머리 로컬 x = 월드 위.
    tray = ET.SubElement(head, "body", name="carry_tray",
                         pos=fmt(HEAD_TOP_LOCAL + np.array([2 * TRAY_HALF_T, 0, 0])),
                         quat="0.707107 0 0.707107 0")
    ET.SubElement(tray, "inertial", pos=fmt([0, 0, -TRAY_HALF_T]), mass=str(TRAY_MASS),
                  diaginertia="2.4e-5 2.4e-5 4.8e-5")
    ET.SubElement(tray, "geom", name="carry_tray", type="box",
                  pos=fmt([0, 0, -TRAY_HALF_T]), size=fmt([TRAY_HALF, TRAY_HALF, TRAY_HALF_T]),
                  contype="2", conaffinity="0", friction="0.3 0.005 0.0001",
                  rgba="0.85 0.75 0.55 1", group="1")
    ET.SubElement(tray, "site", name="carry_tray", pos="0 0 0", size="0.004")
    ET.indent(tree, "  ")
    robot_file = "open_duck_mini_v2_carry.xml"
    tree.write(XMLDIR / robot_file, encoding="unicode")

    for kind in ("box", "ball"):
        if kind == "box":
            g = (f'<geom name="carry_object" type="box" size="{OBJ_HALF} {OBJ_HALF} {OBJ_HALF}" '
                 f'mass="{OBJ_MASS}" friction="{OBJ_FRICTION} 0.005 0.0001" rgba="0.9 0.25 0.2 1"'
                 f' contype="0" conaffinity="3" condim="3"/>')
        else:
            # 공은 구름 마찰이 없으면 쟁반이 조금만 기울어도 끝까지 굴러간다 (그게 과제다).
            g = (f'<geom name="carry_object" type="sphere" size="{OBJ_HALF}" '
                 f'mass="{OBJ_MASS}" friction="{OBJ_FRICTION} 0.005 0.0001" rgba="0.2 0.5 0.9 1"'
                 f' contype="0" conaffinity="3" condim="3"/>')
        world = (f'        <body name="carry_object" pos="0 0 1">\n'
                 f'            <freejoint name="carry_object"/>\n'
                 f'            {g}\n'
                 f'        </body>')
        contact = ("    <contact>\n"
                   '        <exclude body1="carry_object" body2="foot_assembly"/>\n'
                   '        <exclude body1="carry_object" body2="foot_assembly_2"/>\n'
                   "    </contact>")

        def fill(m, qpos):
            d = mujoco.MjData(m)
            d.qpos[:] = qpos
            mujoco.mj_kinematics(m, d)
            tb = m.body("carry_tray").id
            R = d.xmat[tb].reshape(3, 3)
            a = m.joint("carry_object").qposadr[0]
            qpos[a:a + 3] = d.xpos[tb] + R[:, 2] * (OBJ_HALF + 0.001)
            qpos[a + 3:a + 7] = d.xquat[tb]
            return qpos

        m = write_scene(robot_file, XMLDIR / f"scene_carry_{kind}.xml", world, contact, fill)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        tb = m.body("carry_tray").id
        print(f"    쟁반 윗면 월드 {np.round(d.xpos[tb], 4)}  z축 {np.round(d.xmat[tb].reshape(3, 3)[:, 2], 3)}"
              f"  물건 {np.round(d.xpos[m.body('carry_object').id], 4)}")


def make_skate():
    m0 = load(SCENE_SRC.read_text(encoding="utf-8"))
    d0 = mujoco.MjData(m0)
    mujoco.mj_resetDataKeyframe(m0, d0, 0)
    mujoco.mj_forward(m0, d0)

    tree = ET.parse(ROBOT_SRC)
    root = tree.getroot()
    bodies = bodies_by_name(root)
    wheel_bodies = {}
    for side, foot in (("left", "foot_assembly"), ("right", "foot_assembly_2")):
        lo, hi = sole_frame(m0, d0, foot)
        # 발 로컬 축: x = 앞(월드 x), y = 위(월드 z), z = 옆(월드 -y). headbox 실측과 같다.
        R = d0.xmat[m0.body(foot).id].reshape(3, 3)
        assert R[2, 1] > 0.99, f"{foot}: 로컬 y 가 위가 아니다 {R[:, 1]}"
        y_c = lo[1] - WHEEL_GAP - WHEEL_R
        z_c = 0.5 * (lo[2] + hi[2])
        span = hi[0] - lo[0]
        xs = np.linspace(lo[0] + WHEEL_R + 0.002, hi[0] - WHEEL_R - 0.002, N_WHEELS)
        fb = bodies[foot]
        frame = ET.SubElement(fb, "body", name=f"skate_frame_{side}",
                              pos=fmt([0.5 * (lo[0] + hi[0]), lo[1] - WHEEL_GAP / 2, z_c]))
        ET.SubElement(frame, "inertial", pos="0 0 0", mass=str(FRAME_MASS),
                      diaginertia="2e-6 2e-5 2e-5")
        ET.SubElement(frame, "geom", type="box", contype="0", conaffinity="0", group="1",
                      size=fmt([span / 2, WHEEL_GAP / 2, 0.006]), rgba="0.15 0.15 0.15 1")
        names = []
        for i, x in enumerate(xs):
            wb = ET.SubElement(fb, "body", name=f"skate_wheel_{side}_{i}", pos=fmt([x, y_c, z_c]))
            ET.SubElement(wb, "joint", name=f"skate_wheel_{side}_{i}", type="hinge", axis="0 0 1",
                          damping=str(WHEEL_DAMPING), frictionloss="0", armature="1e-6",
                          limited="false")
            ET.SubElement(wb, "geom", name=f"skate_wheel_{side}_{i}", type="sphere",
                          size=str(WHEEL_R), mass=str(WHEEL_MASS), contype="0", conaffinity="1",
                          condim="3", friction="0.9 0.005 0.0001", rgba="0.95 0.8 0.1 1",
                          group="1")
            names.append(f"skate_wheel_{side}_{i}")
        wheel_bodies[side] = names
        print(f"  {side}: 밑창 길이 {span * 100:.1f} cm, 바퀴 x {np.round(xs * 100, 1)} cm")
    ET.indent(tree, "  ")
    robot_file = "open_duck_mini_v2_skate.xml"
    tree.write(XMLDIR / robot_file, encoding="unicode")

    # 왼발 바퀴 ↔ 오른발(및 오른발 바퀴) 충돌 제외 — 발끼리 엉키는 건 원래도 안 본다.
    ex = []
    for wl in wheel_bodies["left"]:
        ex.append(f'        <exclude body1="{wl}" body2="foot_assembly_2"/>')
        for wr in wheel_bodies["right"]:
            ex.append(f'        <exclude body1="{wl}" body2="{wr}"/>')
    for wr in wheel_bodies["right"]:
        ex.append(f'        <exclude body1="{wr}" body2="foot_assembly"/>')
    contact = "    <contact>\n" + "\n".join(ex) + "\n    </contact>"

    raise_z = 2 * WHEEL_R + WHEEL_GAP

    def fill(m, qpos):
        qpos[2] += raise_z
        return qpos

    m = write_scene(robot_file, XMLDIR / "scene_skate.xml", None, contact, fill)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    low = min(d.geom_xpos[m.geom(n).id][2] - WHEEL_R
              for side in wheel_bodies for n in wheel_bodies[side])
    print(f"    home 에서 바퀴 최저점 z {low * 1000:+.1f} mm (원본 밑창은 -16 mm 로 박혀 시작한다)")


if __name__ == "__main__":
    which = sys.argv[1:] or ["carry", "skate"]
    if "carry" in which:
        print("[carry]")
        make_carry()
    if "skate" in which:
        print("[skate]")
        make_skate()
