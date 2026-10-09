r"""벽에 부딪히기 · 경사로 오르기 학습용 씬 (obstacle). make_addon_xml.py 와 같은 방식으로 복사본을 만든다.

원본 open_duck_mini_v2_backlash.xml / scene_flat_terrain_backlash.xml 은 건드리지 않는다.

로봇 — open_duck_mini_v2_backlash_collision.xml (몸통·골반·정강이·머리에 상자 8개) 을 복사하고
    그 상자들만 contype 2 / conaffinity 0 으로 바꾼다. 그러면 상자는 **벽하고만** 부딪힌다:
    바닥·경사로(contype 1, conaffinity 0)와도, 자기들끼리도 안 부딪힌다 (걷기 학습과 같은 계산량에 가깝게).
    발(밑창 메시)은 원래대로 contype 1 / conaffinity 1 — 바닥과 경사로를 밟는다.

장애물 — 둘 다 mocap body 라서 학습 환경이 에피소드마다 옮긴다 (obstacle.py).
    obst_ramp : **무한 평면**. 기울여서 바닥과 만나는 선이 경사로 시작점이 된다. 평면-메시 충돌은
                바닥과 같은 비용이라 상자 경사로보다 싸다. 안 쓸 때는 z=-1 로 내려 둔다.
    obst_wall : 상자 (두께 10 cm, 폭 2 m, 높이 60 cm). conaffinity 2 라 몸의 상자들과만 부딪힌다
                (발 메시-상자 충돌은 MJX 에서 비싸서 뺐다. 정강이 상자가 발목 위를 덮는다).
                안 쓸 때는 z=-5.

사용:  .venv\Scripts\python.exe make_obstacle_xml.py
"""
import sys

import make_addon_xml as ma

ROBOT_SRC = ma.XMLDIR / "open_duck_mini_v2_backlash_collision.xml"
ROBOT_DST = "open_duck_mini_v2_obstacle.xml"
SCENE_DST = ma.XMLDIR / "scene_obst_train.xml"

WALL_HALF = (0.05, 1.0, 0.3)

WORLD = f"""
        <body name="obst_ramp" mocap="true" pos="0 0 -1">
            <geom name="obst_ramp" type="plane" size="0 0 0.01" contype="1" conaffinity="0"
                priority="1" friction="0.6" condim="3" rgba="0.55 0.65 0.55 1"/>
        </body>
        <body name="obst_wall" mocap="true" pos="0 0 -5">
            <geom name="obst_wall" type="box" size="{WALL_HALF[0]} {WALL_HALF[1]} {WALL_HALF[2]}"
                contype="0" conaffinity="2" rgba="0.7 0.5 0.4 1"/>
        </body>"""


def make():
    text = ROBOT_SRC.read_text(encoding="utf-8")
    n = 0
    out = []
    for line in text.splitlines(keepends=True):
        if '_ground_collision"' in line and "<geom" in line:
            line = line.replace('class="collision"', 'class="collision" contype="2" conaffinity="0"')
            n += 1
        out.append(line)
    assert n == 8, f"collision boxes {n} != 8"
    (ma.XMLDIR / ROBOT_DST).write_text("".join(out), encoding="utf-8")
    print(f"[obstacle] {ROBOT_DST}: 몸 상자 {n}개 contype 2 / conaffinity 0")
    m = ma.write_scene(ROBOT_DST, SCENE_DST, extra_world=WORLD)
    for g in ["floor", "obst_ramp", "obst_wall", "left_foot_bottom_tpu", "trunk_assembly_ground_collision"]:
        i = m.geom(g).id
        print(f"    {g:34s} contype {m.geom_contype[i]} conaffinity {m.geom_conaffinity[i]}")
    print(f"    mocap bodies {m.nmocap}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    make()
