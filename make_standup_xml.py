"""기립(standup) 학습용 모델 생성.

원본 open_duck_mini_v2.xml 은 충돌 geom 이 발바닥 TPU 2개뿐이라, 넘어지면 몸이
바닥을 통과한다. 기립을 학습하려면 밀어낼 바닥이 있어야 하므로 몸통/머리/다리에
바닥 전용 충돌 박스를 추가한다.

- 박스 크기는 각 body 의 시각 메시 정점 AABB 에서 실측 (SHRINK 만큼 축소)
- 로봇 내부 body 쌍은 <contact><exclude> 로 전부 제외 → 바닥하고만 충돌
- 원본 파일은 수정하지 않는다 (UBAI 학습이 원본을 사용 중)

사용:  python make_standup_xml.py
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, mujoco, sys

XMLDIR = Path("Open_Duck_Playground/playground/open_duck_mini_v2/xmls")
SRC = XMLDIR / "open_duck_mini_v2.xml"
DST = XMLDIR / "open_duck_mini_v2_standup.xml"
SCENE_SRC = XMLDIR / "scene_flat_terrain.xml"
SCENE_DST = XMLDIR / "scene_standup.xml"

# 충돌 박스를 붙일 body
TARGETS = [
    "trunk_assembly", "head_assembly",
    "knee_and_ankle_assembly", "knee_and_ankle_assembly_2",
    "knee_and_ankle_assembly_3", "knee_and_ankle_assembly_4",
    "left_roll_to_pitch_assembly", "right_roll_to_pitch_assembly",
]
# AABB 계산에서 뺄 메시 (가늘고 길어서 범위를 왜곡)
SKIP_MESH = {"antenna", "left_antenna_holder", "right_antenna_holder"}
SHRINK = 0.93

sys.path.insert(0, "Open_Duck_Playground")
from playground.open_duck_mini_v2 import base
from etils import epath

m = mujoco.MjModel.from_xml_string(epath.Path(str(SCENE_SRC)).read_text(), assets=base.get_assets())

def body_box(bname):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
    if bid < 0: return None
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        mid = m.geom_dataid[g]
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) in SKIP_MESH: continue
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[va:va+vn].reshape(-1, 3)
        R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
        pts.append(v @ R.T + m.geom_pos[g])
    if not pts: return None
    P = np.vstack(pts); lo, hi = P.min(0), P.max(0)
    return (lo + hi) / 2, (hi - lo) / 2 * SHRINK

tree = ET.parse(SRC); root = tree.getroot()
bodies = {}
def walk(e):
    for c in list(e):
        if c.tag == "body": bodies[c.get("name")] = c
        walk(c)
walk(root)

added = []
for bname in TARGETS:
    r = body_box(bname)
    if r is None:
        print(f"  건너뜀 (메시 없음): {bname}"); continue
    c, h = r
    g = ET.SubElement(bodies[bname], "geom")
    g.set("type", "box"); g.set("class", "collision")
    g.set("pos", " ".join(f"{x:.5f}" for x in c))
    g.set("size", " ".join(f"{x:.5f}" for x in h))
    g.set("name", f"{bname}_ground_collision")
    added.append(bname)
    print(f"  {bname:<30} 크기 {np.round(h*2*100,1)} cm")

# 자기충돌 처리.
#
# 처음에는 로봇 내부 쌍을 전부 제외했는데, 그러면 다리가 몸통을 그대로 통과한다.
# 강화학습은 그 허점을 찾아내서 현실에 없는 자세로 서는 정책을 만든다 (실제로 그랬다).
#
# 그래서 "원래 겹쳐 있는 쌍"만 제외한다. 충돌 박스는 메시 AABB 근사라 관절 주변에서
# 서로 파고들어 있는 경우가 있는데, 그건 물리적 접촉이 아니라 근사 오차다. 이걸 그냥
# 두면 시뮬이 그 지점에서 계속 밀어내며 폭발한다.
#
# 판별법: 제외 없는 모델을 만들어 home 자세와 무작위 자세 여러 개에서 접촉을 조사해,
# 거의 항상 닿아 있는 쌍만 근사 오차로 보고 제외한다. 가끔 닿는 쌍은 진짜 충돌이므로 남긴다.
# 조사용으로 exclude 없는 모델을 먼저 디스크에 쓴다 (중첩 <mujoco> 는 불가능하므로).
ET.indent(tree, space="  ")
tree.write(DST, encoding="unicode", xml_declaration=False)
SCENE_DST.write_text(
    SCENE_SRC.read_text().replace(
        '<include file="open_duck_mini_v2.xml"/>',
        '<include file="open_duck_mini_v2_standup.xml"/>'), encoding="utf-8")
pm = mujoco.MjModel.from_xml_string(
    epath.Path(str(SCENE_DST)).read_text(), assets=base.get_assets())
pd = mujoco.MjData(pm)
kid = mujoco.mj_name2id(pm, mujoco.mjtObj.mjOBJ_KEY, "home")

N_PROBE = 60
counts = {}
rng = np.random.default_rng(0)
for i in range(N_PROBE):
    mujoco.mj_resetDataKeyframe(pm, pd, kid)
    if i > 0:  # 0번은 home 자세 그대로
        for j in range(pm.njnt):
            if pm.jnt_type[j] == 0 or pm.jnt_limited[j] == 0: continue
            lo, hi = pm.jnt_range[j]
            pd.qpos[pm.jnt_qposadr[j]] = rng.uniform(lo, hi)
    mujoco.mj_forward(pm, pd)
    hit = set()
    for c in range(pd.ncon):
        b1 = pm.geom_bodyid[pd.contact[c].geom1]
        b2 = pm.geom_bodyid[pd.contact[c].geom2]
        n1 = mujoco.mj_id2name(pm, mujoco.mjtObj.mjOBJ_BODY, b1)
        n2 = mujoco.mj_id2name(pm, mujoco.mjtObj.mjOBJ_BODY, b2)
        if "floor" in (n1, n2) or n1 == n2: continue
        hit.add(frozenset((n1, n2)))
    for k in hit:
        counts[k] = counts.get(k, 0) + 1

ALWAYS = 0.9  # 90% 이상 닿아 있으면 근사 오차로 간주
seen = {k for k, v in counts.items() if v >= N_PROBE * ALWAYS}
contact = ET.SubElement(root, "contact")
for k in sorted(seen, key=lambda x: sorted(x)):
    a, b = sorted(k)
    e = ET.SubElement(contact, "exclude"); e.set("body1", a); e.set("body2", b)

print(f"  자세 {N_PROBE}개 조사: 접촉 발생 쌍 {len(counts)}개")
for k, v in sorted(counts.items(), key=lambda x: -x[1]):
    a, b = sorted(k)
    mark = "제외(근사오차)" if k in seen else "유지(실제 충돌)"
    print(f"    {a:<28} <-> {b:<28} {v:3d}/{N_PROBE}  {mark}")

ET.indent(tree, space="  ")
tree.write(DST, encoding="unicode", xml_declaration=False)
SCENE_DST.write_text(
    SCENE_SRC.read_text().replace(
        '<include file="open_duck_mini_v2.xml"/>',
        '<include file="open_duck_mini_v2_standup.xml"/>'), encoding="utf-8")
print(f"\n충돌 박스 {len(added)}개, 자기충돌 제외 {len(seen)}쌍")
print("생성:", DST); print("생성:", SCENE_DST)
