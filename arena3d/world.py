"""Build a two-character MuJoCo world from the CMU humanoid model.

`build_world(boss_scale, weapon)` returns an MjModel with
  * boss: CMU humanoid scaled by `boss_scale`, prefix "b_", free root, a weapon capsule in its right hand;
  * player: CMU humanoid, prefix "p_", free root;
  * floor, lights, a third-person camera.
Contact filter: weapon (contype 2) and boss body (contype 4) touch only player geoms (conaffinity 6).
Nothing else collides; motion is kinematic (set qpos, call mj_forward / mj_collision).
"""
from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ASSET = None
for _p in [os.path.join(os.path.dirname(__file__), "assets", "humanoid_CMU_V2020.xml")]:
    if os.path.exists(_p):
        ASSET = _p
if ASSET is None:
    import dm_control  # only for the asset file
    ASSET = os.path.join(os.path.dirname(dm_control.__file__), "locomotion/walkers/assets/humanoid_CMU_V2020.xml")

WEAPONS = {  # name: (length m, radius m, grip offset along hand)  in boss units before scaling
    "sword": (1.0, 0.035, 0.05),
    "greatsword": (1.5, 0.05, 0.05),
    "club": (0.9, 0.08, 0.05),
    "fist": (0.0, 0.0, 0.0),
}


def _scale_attr(el, name, s):
    v = el.get(name)
    if v is None:
        return
    el.set(name, " ".join(f"{float(x) * s:.6g}" for x in v.split()))


def _humanoid(prefix, scale, rgba, contype, conaffinity):
    tree = ET.parse(ASSET)
    root = tree.getroot()
    body = root.find("worldbody").find("body")
    body = copy.deepcopy(body)
    for el in body.iter():
        if el.get("name"):
            el.set("name", prefix + el.get("name"))
        if el.tag == "geom":
            for a in ("pos", "size", "fromto"):
                _scale_attr(el, a, scale)
            el.set("rgba", rgba)
            el.set("contype", str(contype))
            el.set("conaffinity", str(conaffinity))
            el.set("group", "1")
            el.attrib.pop("material", None)
        elif el.tag in ("body", "site", "camera"):
            _scale_attr(el, "pos", scale)
            if el.tag == "site":
                _scale_attr(el, "size", scale)
        if el.tag == "joint":
            pass
    for el in list(body):
        if el.tag in ("camera",):
            body.remove(el)
    body.insert(0, ET.Element("freejoint", {"name": prefix + "root"}))
    return body, root


def build_world(boss_scale=1.6, weapon="greatsword", width=128, height=128):
    b, src = _humanoid("b_", boss_scale, "0.35 0.3 0.4 1", 4, 8)
    p, _ = _humanoid("p_", 1.0, "0.75 0.6 0.4 1", 0, 6)
    # player: straight sword in the right fist (contype 8 touches boss volumes), shield on the left forearm
    ph = [e for e in p.iter("body") if e.get("name") == "p_rhand"][0]
    pw = ET.SubElement(ph, "body", {"name": "p_weapon"})
    ET.SubElement(pw, "geom", {"name": "p_blade", "type": "capsule", "fromto": "-0.05 -0.06 0 -0.95 -0.06 0",
                               "size": "0.025", "rgba": "0.8 0.8 0.85 1", "contype": "8", "conaffinity": "0",
                               "group": "1"})
    ET.SubElement(pw, "site", {"name": "p_tip", "pos": "-0.95 -0.06 0", "size": "0.02"})
    la = [e for e in p.iter("body") if e.get("name") == "p_lradius"][0]
    ET.SubElement(la, "geom", {"name": "p_shield", "type": "box", "pos": "0 -0.12 0.06", "size": "0.22 0.2 0.015",
                               "rgba": "0.45 0.35 0.2 1", "contype": "0", "conaffinity": "2", "group": "1"})
    L, R, grip = WEAPONS[weapon]
    if L > 0:
        hand = [e for e in b.iter("body") if e.get("name") == "b_rhand"][0]
        wb = ET.SubElement(hand, "body", {"name": "b_weapon", "pos": "0 0 0"})
        # gripped in the fist (fingers run along local -y); the blade points forward of the fist (-x in the CMU hand frame)
        s = boss_scale
        gy = -0.06 * s
        ET.SubElement(wb, "geom", {"name": "b_blade", "type": "capsule",
                                    "fromto": f"{-grip * s:.4f} {gy:.4f} 0 {-(grip + L) * s:.4f} {gy:.4f} 0",
                                    "size": f"{R * s:.4f}", "rgba": "0.85 0.85 0.9 1",
                                    "contype": "2", "conaffinity": "0", "group": "1"})
        ET.SubElement(wb, "site", {"name": "b_tip", "pos": f"{-(grip + L) * s:.4f} {gy:.4f} 0", "size": "0.02"})
    m = ET.Element("mujoco", {"model": "arena3d"})
    ET.SubElement(m, "compiler", {"angle": "radian"})
    ET.SubElement(m, "option", {"timestep": "0.0075"})
    vis = ET.SubElement(m, "visual")
    ET.SubElement(vis, "global", {"offwidth": str(max(width, 640)), "offheight": str(max(height, 480))})
    ET.SubElement(vis, "quality", {"shadowsize": "1024"})
    asset = ET.SubElement(m, "asset")
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d", "builtin": "checker", "rgb1": ".25 .24 .22",
                                     "rgb2": ".3 .29 .27", "width": "256", "height": "256"})
    ET.SubElement(asset, "material", {"name": "self", "rgba": ".7 .5 .3 1"})
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid", "texrepeat": "8 8"})
    ET.SubElement(asset, "texture", {"name": "sky", "type": "skybox", "builtin": "gradient", "rgb1": ".2 .2 .25",
                                     "rgb2": "0 0 0", "width": "64", "height": "64"})
    m.append(copy.deepcopy(src.find("default")))
    wb = ET.SubElement(m, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 8", "dir": "0 0 -1", "diffuse": ".8 .8 .8", "castshadow": "true"})
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "size": "20 20 .1", "material": "grid",
                               "contype": "0", "conaffinity": "0"})
    ET.SubElement(wb, "camera", {"name": "third", "mode": "fixed", "pos": "0 -4 2", "xyaxes": "1 0 0 0 0.4 1"})
    wb.append(b)
    wb.append(p)
    xml = ET.tostring(m, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml), xml
