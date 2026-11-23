#!/usr/bin/env python3
"""Generate the tea-bag scene for stretch_mujoco.

Writes tea_bag.png / tea_tag.png / paper_cup.png into <models>/assets and
rewrites <models>/scene.xml with:
  * a pillow-shaped tea bag (stitched border, leaf shadow) with a cotton string
    and a printed paper tag, all on ONE rigid body (lift the tag -> bag follows)
  * a hollow white paper cup with a cardboard sleeve (open top, bag can drop in)
  * a bright orange rectangular block (the grasp target): 1 x 1 in square
    base, 3 in tall, standing upright (2x the real 0.5 x 0.5 x 1.5 in block so it
    is big enough to detect in the low-res sim cameras)

Usage:  python3 tools/make_scene.py [models_dir]
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODELS = Path.home() / 'repos/stretch_mujoco/stretch_mujoco/models'

# ---- geometry (metres) ----
BAG_HALF = (0.04, 0.03, 0.006)          # bag half sizes x, y, z
TAG_HALF = (0.032, 0.022, 0.0015)       # paper tag
# string lies flat with a gentle S-bend (slack), pinned to bag AND tag
STRING_PTS = [(0.040, 0, 0.005), (0.058, 0.006, 0.003), (0.076, -0.006, 0.003),
              (0.094, 0.005, 0.003), (0.110, 0, 0.003)]
TAG_CENTER = (0.110 + TAG_HALF[0], 0, TAG_HALF[2])

BAG_ROT, TAG_ROT = 3, 2   # np.rot90 turns so photo features land on the right box ends
CUP_R, CUP_HALF_H, CUP_WALL, CUP_SIDES = 0.045, 0.05, 0.0015, 20
TABLE_TOP = 0.48
IN = 0.0254
BLOCK_HALF = (0.5 * IN, 0.5 * IN, 1.5 * IN)     # 1 x 1 in base, 3 in tall (2x real size)
BLOCK_XY = (-0.22, -0.56)   # ~20 cm in x from the gripper at startup (x = -0.02), still in head-cam view


def _noise(rng, h, w, sigma):
    n = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), sigma)
    return (n - n.min()) / (n.max() - n.min() + 1e-6)


def tea_bag_texture(w=320, h=240):
    """Filter-paper pillow bag: crimped edges, staple fold at the +x end, leaves inside."""
    rng = np.random.default_rng(1)
    paper = np.array((150, 182, 206), np.float32)                 # BGR
    img = np.tile(paper, (h, w, 1))
    img *= (0.93 + 0.1 * _noise(rng, h, w, 0.8))[..., None]       # paper fibres
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # tea leaves showing through the mesh: mottled brown blob in the pillow centre
    cx, cy = w * 0.45, h * 0.5
    blob = np.clip(1.2 - (((xx - cx) / (w * 0.40)) ** 2 + ((yy - cy) / (h * 0.42)) ** 2), 0, 1)
    leaf = np.clip((_noise(rng, h, w, 2) - 0.25) * 2.5, 0, 1) * blob
    img = img * (1 - 0.9 * leaf[..., None]) + np.array((20, 40, 65), np.float32) * 0.9 * leaf[..., None]
    # pillow shading: puffy centre highlight, darker edges
    d = ((xx - cx) / (w * 0.6)) ** 2 + ((yy - cy) / (h * 0.6)) ** 2
    img *= (1.08 - 0.30 * np.clip(d, 0, 1.4))[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    # crimped (zig-zag sealed) edges on three sides
    m = 16
    for x in range(0, w, 6):
        c = (170, 190, 200) if (x // 6) % 2 else (225, 240, 247)
        cv2.line(img, (x, 0), (x, m), c, 3)
        cv2.line(img, (x, h - m), (x, h), c, 3)
    for y in range(0, h, 6):
        c = (170, 190, 200) if (y // 6) % 2 else (225, 240, 247)
        cv2.line(img, (0, y), (m, y), c, 3)
    cv2.line(img, (0, m), (w, m), (150, 170, 182), 1)
    cv2.line(img, (0, h - m), (w, h - m), (150, 170, 182), 1)
    cv2.line(img, (m, 0), (m, h), (150, 170, 182), 1)
    # folded-over top strip with two metal staples (string attaches here)
    fx = w - 62
    cv2.rectangle(img, (fx, 0), (w, h), (196, 214, 224), -1)
    cv2.line(img, (fx, 0), (fx, h), (120, 140, 152), 2)
    cv2.line(img, (fx + 4, 0), (fx + 4, h), (235, 247, 252), 1)
    for sy in (h // 3, 2 * h // 3):
        cv2.rectangle(img, (fx + 22, sy - 4), (fx + 44, sy + 4), (170, 170, 175), -1)
        cv2.rectangle(img, (fx + 22, sy - 4), (fx + 44, sy + 4), (90, 90, 95), 1)
    return img


def tea_tag_texture(w=256, h=176):
    """Printed paper tea tag: burgundy header, leaf illustration, text, eyelet hole."""
    img = np.full((h, w, 3), (232, 242, 248), np.uint8)
    cv2.rectangle(img, (0, 0), (w, 54), (60, 40, 120), -1)             # burgundy header
    cv2.putText(img, 'ROYAL', (62, 24), cv2.FONT_HERSHEY_DUPLEX, 0.7, (235, 245, 250), 1, cv2.LINE_AA)
    cv2.putText(img, 'TEA CO.', (62, 46), cv2.FONT_HERSHEY_DUPLEX, 0.7, (235, 245, 250), 1, cv2.LINE_AA)
    cv2.circle(img, (30, 27), 12, (225, 235, 240), -1)                  # eyelet
    cv2.circle(img, (30, 27), 12, (130, 130, 130), 2)
    cv2.circle(img, (30, 27), 6, (60, 40, 120), -1)
    # leaf illustration
    cv2.ellipse(img, (w // 2 - 40, 112), (46, 24), -30, 0, 360, (70, 140, 60), -1)
    cv2.line(img, (w // 2 - 78, 130), (w // 2 - 4, 92), (40, 90, 35), 2)
    for k in range(-2, 3):
        cv2.line(img, (w // 2 - 41 + 9 * k, 111 - 5 * k), (w // 2 - 41 + 9 * k + 10, 100 - 5 * k), (40, 90, 35), 1)
    cv2.putText(img, 'English', (w // 2 + 14, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 40, 120), 2, cv2.LINE_AA)
    cv2.putText(img, 'Breakfast', (w // 2 + 4, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 40, 120), 2, cv2.LINE_AA)
    cv2.putText(img, 'BLACK TEA', (w // 2 + 8, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (170, 180, 185), 2)
    return img


def paper_cup_texture(w=256, h=128):
    img = np.full((h, w, 3), (250, 250, 250), np.uint8)
    cv2.rectangle(img, (0, 0), (w, 22), (200, 200, 200), -1)
    return img


def fmt(v):
    return ' '.join(f'{x:.5g}' for x in v)


def cup_xml():
    """Hollow cup: N wall slabs + bottom, plus a sleeve (visual only)."""
    out = ['    <body name="cup" pos="0.05 -.65 .53" euler="0 0 2.4">',
           '      <freejoint/>']
    slab_w = CUP_R * math.tan(math.pi / CUP_SIDES) * 1.05
    for i in range(CUP_SIDES):
        a = 2 * math.pi * i / CUP_SIDES
        pos = (CUP_R * math.cos(a), CUP_R * math.sin(a), 0)
        out.append(
            f'      <geom type="box" size="{CUP_WALL} {slab_w:.5f} {CUP_HALF_H}" '
            f'pos="{fmt(pos)}" euler="0 0 {a:.5f}" mass=".004" friction="1 .1 .1" '
            f'material="paper_cup"/>')
    out.append(f'      <geom type="cylinder" size="{CUP_R} .002" pos="0 0 {-CUP_HALF_H + 0.002}" '
               f'mass=".02" friction="1 .1 .1" material="paper_cup"/>')
    # cardboard sleeve ring (collision-free, just looks like a coffee sleeve)
    for i in range(CUP_SIDES):
        a = 2 * math.pi * i / CUP_SIDES
        r = CUP_R + 0.002
        pos = (r * math.cos(a), r * math.sin(a), -0.005)
        out.append(
            f'      <geom type="box" size="0.001 {slab_w:.5f} 0.018" pos="{fmt(pos)}" '
            f'euler="0 0 {a:.5f}" contype="0" conaffinity="0" mass="0" rgba=".62 .45 .28 1"/>')
    out.append('    </body>')
    return '\n'.join(out)


def scene_xml():
    pts = ' '.join(fmt(p) for p in STRING_PTS)
    elems = ' '.join(f'{i} {i + 1}' for i in range(len(STRING_PTS) - 1))
    return f'''<mujoco model="stretch scene">
  <include file="stretch.xml"/>
  <include file="docking_station.xml"/>

  <statistic center="0 0 .75" extent="1.2" meansize="0.05"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-120" elevation="-20"/>
  </visual>

  <asset>
    <material name="floor" rgba=".1 .1 .1 1" reflectance="0.1"/>
    <texture type="2d" name="wood" file="wood.png"/>
    <material name="wood" texture="wood"/>
    <material name="object" rgba=".2 .2 .2 1"/>
    <texture type="skybox" builtin="gradient" rgb1="0.44 0.80 1.00" rgb2="1 1 1" width="512" height="3072"/>
    <texture type="2d" name="tea_bag" file="tea_bag.png"/>
    <material name="tea_bag" texture="tea_bag" rgba=".72 .72 .72 1"/>
    <texture type="2d" name="tea_tag" file="tea_tag.png"/>
    <material name="tea_tag" texture="tea_tag" rgba=".85 .85 .85 1"/>
    <material name="orange_block" rgba="1 .42 0 1" specular=".2" shininess=".3"/>
    <texture type="2d" name="paper_cup" file="paper_cup.png"/>
    <material name="paper_cup" texture="paper_cup" rgba="1 1 1 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floor"/>
    <body name="table" pos="0 -1 .24">
      <geom type="box" size=".6 .5 .24" mass="1" material="wood"/>
    </body>

    <!-- Tea bag: pillow bag + string (flexcomp, pinned to bag) + paper tag.
         Bag and tag are geoms of one body so lifting the tag lifts the bag. -->
    <body name="tea_bag" pos=".12 -.55 .49">
      <freejoint/>
      <geom name="bag" type="box" size="{fmt(BAG_HALF)}" mass=".02" friction=".8 .1 .1"
            material="tea_bag"/>
      <geom name="tag" type="box" size="{fmt(TAG_HALF)}" pos="{fmt(TAG_CENTER)}"
            mass=".005" friction="1 .1 .1" material="tea_tag"/>
      <flexcomp name="tea_string" type="direct" dim="1" mass=".004" radius=".0012"
                rgba=".95 .95 .9 1" point="{pts}" element="{elems}">
        <pin id="0 {len(STRING_PTS) - 1}"/>
        <edge equality="true" stiffness="100" damping="1"/>
      </flexcomp>
    </body>

    <!-- Orange block: the grasp target (1 x 1 in base, 3 in tall, upright; 2x real size) -->
    <body name="orange_block" pos="{fmt((*BLOCK_XY, TABLE_TOP + BLOCK_HALF[2]))}">
      <freejoint/>
      <geom name="orange_block" type="box" size="{fmt(BLOCK_HALF)}" mass=".03"
            friction="1.5 .05 .01" condim="4" material="orange_block"/>
    </body>

    <!-- Paper cup (hollow, open top) -->
{cup_xml()}
  </worldbody>
</mujoco>
'''


def main():
    models = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODELS
    assets = models / 'assets'
    here = Path(__file__).parent / 'assets'
    # real photo crops (CC0, see assets/LICENSE.txt) when present, else procedural
    bag = cv2.imread(str(here / 'tea_bag_photo.png'))
    tag = cv2.imread(str(here / 'tea_tag_photo.png'))
    bag = tea_bag_texture() if bag is None else np.rot90(bag, BAG_ROT)
    tag = tea_tag_texture() if tag is None else np.rot90(tag, TAG_ROT)
    cv2.imwrite(str(assets / 'tea_bag.png'), np.ascontiguousarray(bag))
    cv2.imwrite(str(assets / 'tea_tag.png'), np.ascontiguousarray(tag))
    cv2.imwrite(str(assets / 'paper_cup.png'), paper_cup_texture())
    (models / 'scene.xml').write_text(scene_xml())
    print(f'wrote scene + textures to {models}')


if __name__ == '__main__':
    main()
