#!/usr/bin/env python3
"""
交互式 3D quadtree 可视化 (Three.js)：
  上层：生成图片 + 8×8 grid (LOD 3)
  下层：生成图片 + 16×16 grid (LOD 4)，下层长宽为上层 2 倍
  连线表示 parent → child 关系。支持旋转/缩放/平移，支持导出 PNG。

用法 (只需指定图片，自动找 plan):
  python tree_planning/visualize_plan_3d.py \
    --image tree_planning/gen_out/generation/class_001_goldfish__Carassius_auratus_plan_0.png

  也可手动指定 plan:
  python tree_planning/visualize_plan_3d.py \
    --plan tree_planning/plans/class_001_goldfish/plan_0.pkl \
    --image tree_planning/gen_out/generation/class_001_goldfish__Carassius_auratus_plan_0.png
"""
import base64
import glob
import json
import pickle
import re
import argparse
from pathlib import Path

TREE_PLANNING_DIR = Path(__file__).resolve().parent


def _parent_8x8_of_16x16(idx_16):
    r, c = idx_16 // 16, idx_16 % 16
    return (r // 2) * 8 + (c // 2)


def load_image_b64(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def resolve_plan_from_image(image_path: str) -> str:
    """Given an image like class_001_goldfish__Carassius_auratus_plan_0.png,
    find the corresponding plan pkl in tree_planning/plans/."""
    stem = Path(image_path).stem
    m = re.match(r"class_(\d{3})_.*_plan_(\d+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse class_id/plan_idx from image filename: {stem}")
    class_id, plan_idx = m.group(1), m.group(2)
    plans_dir = TREE_PLANNING_DIR / "plans"
    matches = sorted(plans_dir.glob(f"class_{class_id}_*/plan_{plan_idx}.pkl"))
    if not matches:
        raise FileNotFoundError(
            f"No plan found for class_{class_id}_*/plan_{plan_idx}.pkl in {plans_dir}")
    return str(matches[0])


def build_html(plan: dict, image_path: str, title: str = "") -> str:
    patch_indices = plan["patch_indices"]
    if hasattr(patch_indices, "tolist"):
        patch_indices = patch_indices.tolist()

    leaves_16 = list(patch_indices[64:])
    leaves_set = set(leaves_16)
    parents_8 = set(_parent_8x8_of_16x16(idx) for idx in leaves_16)

    # Build parent->children mapping
    parent_children = {}
    for idx in leaves_16:
        p = _parent_8x8_of_16x16(idx)
        parent_children.setdefault(p, []).append(idx)

    img_b64 = load_image_b64(image_path)

    data_json = json.dumps({
        "leaves_16": sorted(leaves_set),
        "parents_8": sorted(parents_8),
        "parent_children": {str(k): v for k, v in parent_children.items()},
        "title": title,
    })

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>{title or "Plan Tree Viz"}</title>
<style>
  body {{ margin:0; overflow:hidden; background:#111; }}
  #info {{ position:absolute; top:10px; left:10px; color:#ccc; font:14px system-ui; pointer-events:none; z-index:1; }}
  #toolbar {{ position:absolute; top:10px; right:10px; z-index:2; display:flex; gap:8px; }}
  #toolbar button {{
    padding:6px 14px; border:1px solid #555; border-radius:4px;
    background:#222; color:#ddd; font:13px system-ui; cursor:pointer;
  }}
  #toolbar button:hover {{ background:#444; }}
</style>
</head><body>
<div id="info">{title}<br><small>drag to rotate &middot; scroll to zoom &middot; right-drag to pan</small></div>
<div id="toolbar">
  <button id="btnPng">Export PNG</button>
  <button id="btnReset">Reset View</button>
</div>
<script type="importmap">
{{"imports":{{"three":"https://esm.sh/three@0.162.0","three/addons/":"https://esm.sh/three@0.162.0/examples/jsm/"}}}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ Line2 }} from 'three/addons/lines/Line2.js';
import {{ LineMaterial }} from 'three/addons/lines/LineMaterial.js';
import {{ LineGeometry }} from 'three/addons/lines/LineGeometry.js';

const DATA = {data_json};
const IMG_B64 = "data:image/png;base64,{img_b64}";

function fatLine(pts, color, width, opacity) {{
  const pos = [];
  for (const p of pts) pos.push(p.x, p.y, p.z);
  const geo = new LineGeometry();
  geo.setPositions(pos);
  const mat = new LineMaterial({{
    color, linewidth: width, transparent: true, opacity,
    resolution: new THREE.Vector2(innerWidth, innerHeight),
  }});
  return new Line2(geo, mat);
}}

// --- Renderer (preserveDrawingBuffer for screenshot) ---
const renderer = new THREE.WebGLRenderer({{antialias:true, preserveDrawingBuffer:true}});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.1, 500);
const INIT_POS = new THREE.Vector3(16, 28, 38);
const INIT_TARGET = new THREE.Vector3(8, 2, 8);
camera.position.copy(INIT_POS);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(INIT_TARGET);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 1.0));

// --- Texture ---
const texLoader = new THREE.TextureLoader();
function makeTex() {{
  const t = texLoader.load(IMG_B64);
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}}

// --- Dimensions ---
const UPPER_SIZE = 16;          // 8x8 grid, cell=2
const LOWER_SIZE = 32;          // 16x16 grid, cell=2  (2x upper)
const UPPER_CELL = 2;
const LOWER_CELL = 2;
const UPPER_Y = 10, LOWER_Y = 0;
const UO = 0;                   // upper origin x/z
const LO = UO + (UPPER_SIZE - LOWER_SIZE) / 2;  // center-aligned => -8

// --- Upper layer: image + 8x8 grid ---
const upperPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(UPPER_SIZE, UPPER_SIZE),
  new THREE.MeshBasicMaterial({{map: makeTex(), side: THREE.DoubleSide}})
);
upperPlane.rotation.x = -Math.PI / 2;
upperPlane.position.set(UO + UPPER_SIZE/2, UPPER_Y, UO + UPPER_SIZE/2);
scene.add(upperPlane);

// 8x8 grid lines (fat)
for (let i = 0; i <= 8; i++) {{
  const x = UO + i * UPPER_CELL;
  scene.add(fatLine([
    new THREE.Vector3(x, UPPER_Y+0.01, UO),
    new THREE.Vector3(x, UPPER_Y+0.01, UO+UPPER_SIZE)], 0xffffff, 2, 0.4));
  scene.add(fatLine([
    new THREE.Vector3(UO, UPPER_Y+0.01, x),
    new THREE.Vector3(UO+UPPER_SIZE, UPPER_Y+0.01, x)], 0xffffff, 2, 0.4));
}}

// Highlight activated 8x8 parents
const hlMat8 = new THREE.MeshBasicMaterial({{color:0xe74c3c, transparent:true, opacity:0.35, side:THREE.DoubleSide}});
for (const p8 of DATA.parents_8) {{
  const r = Math.floor(p8/8), c = p8%8;
  const m = new THREE.Mesh(new THREE.PlaneGeometry(UPPER_CELL, UPPER_CELL), hlMat8);
  m.rotation.x = -Math.PI/2;
  m.position.set(UO + c*UPPER_CELL + UPPER_CELL/2, UPPER_Y+0.02, UO + r*UPPER_CELL + UPPER_CELL/2);
  scene.add(m);
}}

// --- Lower layer: image + 16x16 grid (2x size) ---
const lowerPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(LOWER_SIZE, LOWER_SIZE),
  new THREE.MeshBasicMaterial({{map: makeTex(), side: THREE.DoubleSide}})
);
lowerPlane.rotation.x = -Math.PI / 2;
lowerPlane.position.set(LO + LOWER_SIZE/2, LOWER_Y, LO + LOWER_SIZE/2);
scene.add(lowerPlane);

// 16x16 grid lines (fat)
for (let i = 0; i <= 16; i++) {{
  const x = LO + i * LOWER_CELL;
  scene.add(fatLine([
    new THREE.Vector3(x, LOWER_Y+0.01, LO),
    new THREE.Vector3(x, LOWER_Y+0.01, LO+LOWER_SIZE)], 0xffffff, 2, 0.25));
  scene.add(fatLine([
    new THREE.Vector3(LO, LOWER_Y+0.01, x),
    new THREE.Vector3(LO+LOWER_SIZE, LOWER_Y+0.01, x)], 0xffffff, 2, 0.25));
}}

// Darken entire lower layer, then "punch through" for leaf cells using a canvas mask
const leafSet = new Set(DATA.leaves_16);
const maskCanvas = document.createElement('canvas');
maskCanvas.width = 16; maskCanvas.height = 16;
const mctx = maskCanvas.getContext('2d');
mctx.fillStyle = 'rgba(0,0,0,0.80)';
mctx.fillRect(0, 0, 16, 16);
for (const idx of DATA.leaves_16) {{
  const r = Math.floor(idx/16), c = idx%16;
  mctx.clearRect(c, r, 1, 1);
}}
const maskTex = new THREE.CanvasTexture(maskCanvas);
maskTex.minFilter = THREE.NearestFilter;
maskTex.magFilter = THREE.NearestFilter;
const darkOverlay = new THREE.Mesh(
  new THREE.PlaneGeometry(LOWER_SIZE, LOWER_SIZE),
  new THREE.MeshBasicMaterial({{map: maskTex, transparent: true, side: THREE.DoubleSide}})
);
darkOverlay.rotation.x = -Math.PI/2;
darkOverlay.position.set(LO + LOWER_SIZE/2, LOWER_Y+0.02, LO + LOWER_SIZE/2);
scene.add(darkOverlay);

// Red border for leaf cells (fat)
for (const idx of DATA.leaves_16) {{
  const r = Math.floor(idx/16), c = idx%16;
  const x0 = LO + c*LOWER_CELL, z0 = LO + r*LOWER_CELL;
  const pts = [
    new THREE.Vector3(x0, LOWER_Y+0.03, z0),
    new THREE.Vector3(x0+LOWER_CELL, LOWER_Y+0.03, z0),
    new THREE.Vector3(x0+LOWER_CELL, LOWER_Y+0.03, z0+LOWER_CELL),
    new THREE.Vector3(x0, LOWER_Y+0.03, z0+LOWER_CELL),
    new THREE.Vector3(x0, LOWER_Y+0.03, z0),
  ];
  scene.add(fatLine(pts, 0xe74c3c, 3, 0.8));
}}

// --- Connecting lines: 8x8 parent center -> 16x16 child center (fat) ---
for (const [p8str, children] of Object.entries(DATA.parent_children)) {{
  const p8 = parseInt(p8str);
  const pr = Math.floor(p8/8), pc = p8%8;
  const px = UO + pc*UPPER_CELL + UPPER_CELL/2;
  const pz = UO + pr*UPPER_CELL + UPPER_CELL/2;
  for (const c16 of children) {{
    const cr = Math.floor(c16/16), cc = c16%16;
    const cx = LO + cc*LOWER_CELL + LOWER_CELL/2;
    const cz = LO + cr*LOWER_CELL + LOWER_CELL/2;
    scene.add(fatLine([
      new THREE.Vector3(px, UPPER_Y, pz),
      new THREE.Vector3(cx, LOWER_Y, cz)
    ], 0xe74c3c, 3, 0.5));
  }}
}}

// --- Export PNG ---
document.getElementById('btnPng').addEventListener('click', () => {{
  renderer.render(scene, camera);
  const link = document.createElement('a');
  link.download = 'plan_tree_3d.png';
  link.href = renderer.domElement.toDataURL('image/png');
  link.click();
}});

// --- Reset View ---
document.getElementById('btnReset').addEventListener('click', () => {{
  camera.position.copy(INIT_POS);
  controls.target.copy(INIT_TARGET);
  controls.update();
}});

// --- Resize (update LineMaterial resolution too) ---
window.addEventListener('resize', () => {{
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  scene.traverse(obj => {{
    if (obj.material && obj.material.isLineMaterial) {{
      obj.material.resolution.set(innerWidth, innerHeight);
    }}
  }});
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D quadtree plan visualization")
    parser.add_argument("--image", type=str, required=True,
                        help="Generated image path (e.g. tree_planning/gen_out/generation/class_001_...plan_0.png)")
    parser.add_argument("--plan", type=str, default=None,
                        help="Plan pkl path (auto-resolved from --image if omitted)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output HTML path (default: tree_planning/demo/<image_stem>.html)")
    args = parser.parse_args()

    plan_path = args.plan or resolve_plan_from_image(args.image)
    print(f"Image: {args.image}")
    print(f"Plan:  {plan_path}")

    with open(plan_path, "rb") as f:
        plan = pickle.load(f)
    class_name = plan.get("class_name", "")
    class_id = plan.get("class_id", "?")
    title = f"class {class_id}: {class_name}" if class_name else f"class {class_id}"

    if args.output is None:
        demo_dir = TREE_PLANNING_DIR / "demo"
        out_path = demo_dir / (Path(args.image).stem + ".html")
    else:
        out_path = Path(args.output)

    html = build_html(plan, args.image, title=title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
