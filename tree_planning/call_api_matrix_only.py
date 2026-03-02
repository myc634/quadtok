"""
仿照 call_api.py：调用本地 vLLM API，但 response 只要求 0/1 的 8×8 matrix，无描述文字。
需要先启动: vllm serve Qwen/Qwen3.5-27B --port 8000 ...
"""
import pickle
from openai import OpenAI
import torch
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

MODEL = "Qwen/Qwen3.5-27B"


def chat(messages: list[dict], **kwargs) -> str:
    """发一轮对话请求，返回助手回复内容。"""
    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        **kwargs,
    )
    return r.choices[0].message.content


def parse_matrix_response(reply: str) -> list[int] | None:
    """
    检查 response 是否符合 8×8 的 0/1 矩阵要求。
    要求：恰好 8 行，每行恰好 8 个字符，且只能为 '0' 或 '1'。
    若符合，返回 row-major 的 flatten list（64 个 int）；否则返回 None。
    """
    lines = [line.strip() for line in reply.strip().splitlines()]
    buffer: list[str] = []
    for line in lines:
        if len(line) == 8 and all(c in "01" for c in line):
            buffer.append(line)
            if len(buffer) == 8:
                return [int(c) for row in buffer for c in row]
        else:
            buffer = []
    return None


def matrix_8x8_to_16x16_raster_indices(flat_8x8: list[int]) -> list[int]:
    """
    将 8×8 的 flatten list (row-major) 转为 16×16 上所有「对应 1 的格子」的 rasterization index。
    约定：8×8 的 (i, j) 对应 16×16 的 2×2 块 (2i:2i+2, 2j:2j+2)；
    16×16 的 rasterization index = row * 16 + col，即 row-major，范围 [0, 255]。
    """
    if len(flat_8x8) != 64:
        raise ValueError("flat_8x8 must have 64 elements")
    indices: list[int] = []
    for i in range(8):
        for j in range(8):
            if flat_8x8[i * 8 + j] != 1:
                continue
            # 8×8 (i,j) -> 16×16 的四个格子 (2i,2j), (2i,2j+1), (2i+1,2j), (2i+1,2j+1)
            base = 32 * i + 2 * j
            indices.extend([base, base + 1, base + 16, base + 17])
    return indices


def visualize_8x8(flat_8x8: list[int], file: object | None = None) -> None:
    """
    把 8×8 的 flatten list 画成 8×8 的 emoji 网格：1 → 🟥，0 → 🟦。
    """
    import sys
    out = file if file is not None else sys.stderr
    if len(flat_8x8) != 64:
        return
    for r in range(8):
        line = "".join("🟥" if flat_8x8[r * 8 + c] == 1 else "🟦" for c in range(8))
        print(line, file=out)
    print(file=out)


def visualize_16x16_indices(indices: list[int], file: object | None = None) -> None:
    """
    把 16×16 的 raster index 列表画成 16×16 的 emoji 网格：在 index 里的用 🟥，否则 🟦。
    file 默认 sys.stderr，这样 stdout 仍可管道给下游。
    """
    import sys
    out = file if file is not None else sys.stderr
    seen = set(indices)
    for r in range(16):
        line = "".join("🟥" if r * 16 + c in seen else "🟦" for c in range(16))
        print(line, file=out)
    print(file=out)


# 8×8 在 quadtree 下的 patch 顺序（64 个），与下游一致
PATCH_ORDER_64 = [
    0, 1, 8, 9, 2, 3, 10, 11, 16, 17, 24, 25, 18, 19, 26, 27,
    4, 5, 12, 13, 6, 7, 14, 15, 20, 21, 28, 29, 22, 23, 30, 31,
    32, 33, 40, 41, 34, 35, 42, 43, 48, 49, 56, 57, 50, 51, 58, 59,
    36, 37, 44, 45, 38, 39, 46, 47, 52, 53, 60, 61, 54, 55, 62, 63,
]


def build_plan_dict(
    raster_indices: list[int],
    class_id: int | str = "",
    class_name: str = "",
) -> dict:
    """
    将 raster_indices（16×16 上激活位置的 index 列表）转为下游需要的 dict 格式。
    - class_id, class_name: 便于存储与推理时命名
    - lod_indices: 64 个 3 + len(raster_indices) 个 4
    - patch_indices: PATCH_ORDER_64 + raster_indices
    """
    n_activated = len(raster_indices)
    return {
        "class_id": class_id,
        "class_name": class_name,
        "lod_indices": torch.tensor([3] * 64 + [4] * n_activated, dtype=torch.long),
        "patch_indices": torch.tensor(PATCH_ORDER_64 + raster_indices, dtype=torch.long),
    }


def sanitize_class_name(name: str, max_len: int = 40) -> str:
    """用于目录/文件名：取逗号前部分，空格改下划线，只保留字母数字下划线。"""
    s = name.split(",")[0].strip().replace(" ", "_")
    return "".join(c for c in s if c.isalnum() or c == "_")[:max_len] or "unknown"


def _get_system_prompt_and_base_messages():
    """返回 (system_prompt, base_messages)，base_messages 不含最后一条 user（Category: xxx）。"""
    system_prompt = """
You are given a single object category. Your task is to generate a typical **spatial complexity map** for an assumed ImageNet-style image of that category.

**Semantics (8×8 grid):**
- **1** = this cell is on or near the main object (or its visually complex parts). Prefer object region; mark background as 1 only if clearly complex and fitting the composition.
- **0** = background, sky, plain wall, blur, or other visually simple areas.

**Shape:** 1s should form a readable object silhouette at 8×8 resolution (connected regions, possibly with internal 0s). Blocky but recognizable; can be slightly noisy; activation ratio of 1s ≤ 0.6.

**Output format — ONLY the matrix, nothing else:**
Output exactly 8 lines. Each line has exactly 8 characters: only digits 0 and 1, no spaces, no other text. No description, no explanation.

**Important:** The few-shot examples below only show the required format and possible shape styles (e.g. curved, dome, arc). For the requested category you must always generate a **new** map: use different placement, framing, curvature, or noise—do not copy or repeat the example matrices. Even if the category matches an example (e.g. banana), produce a different valid map for that category.
"""
    banana_matrix = """
00000110
00001100
00111000
01110000
01110010
01100100
01100000
00010000
"""
    parliament_matrix = """
00011000
00111100
01111110
01101110
00111100
01111110
11111101
11011111
"""
    necklace_matrix = """
00001110
00010001
00100000
01000000
01000000
00100000
00010001
00001110
"""
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Category: banana"},
        {"role": "assistant", "content": banana_matrix},
        {"role": "user", "content": "Category: parliament building (e.g. capitol dome)"},
        {"role": "assistant", "content": parliament_matrix},
        {"role": "user", "content": "Category: necklace"},
        {"role": "assistant", "content": necklace_matrix},
    ]
    return system_prompt, base_messages


def _generate_one_plan(
    class_id: int,
    plan_idx: int,
    imagenet_idx2classname: dict,
    base_messages: list,
    plans_dir: Path,
    max_retries: int,
    skip_existing: bool,
) -> tuple[str, int, int]:
    """生成单个 plan 并写入 plans_dir/class_{id:03d}_{name}/plan_{plan_idx}.pkl。返回 ('ok'|'skip'|'fail', class_id, plan_idx)。"""
    category = imagenet_idx2classname[class_id]
    dir_name = f"class_{class_id:03d}_{sanitize_class_name(category)}"
    out_path = plans_dir / dir_name / f"plan_{plan_idx}.pkl"
    if skip_existing and out_path.exists():
        return ("skip", class_id, plan_idx)
    messages = base_messages + [{"role": "user", "content": f"Category: {category}"}]
    for _ in range(max_retries):
        try:
            reply = chat(
                messages=messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception:
            continue
        result = parse_matrix_response(reply)
        if result is not None:
            raster_indices = matrix_8x8_to_16x16_raster_indices(result)
            plan = build_plan_dict(raster_indices, class_id=class_id, class_name=category)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(plan, f)
            return ("ok", class_id, plan_idx)
    return ("fail", class_id, plan_idx)


if __name__ == "__main__":

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from imagenet_classes import imagenet_idx2classname

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Batch: generate plans for all ImageNet classes (5 per class)")
    parser.add_argument("--class-id", type=int, default=None, help="ImageNet class index (0-999), required when not --batch")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries when response format is invalid")
    parser.add_argument("--output", "-o", type=str, default="", help="Save plan dict to this path (single mode)")
    parser.add_argument("--plans-dir", type=str, default=None, help="Batch mode: root dir for plans (default: tree_planning/plans)")
    parser.add_argument("--plans-per-class", type=int, default=5, help="Batch mode: number of plans per class")
    parser.add_argument("--workers", type=int, default=16, help="Batch mode: parallel API workers")
    parser.add_argument("--skip-existing", action="store_true", help="Batch mode: skip class/plan if pkl already exists")
    args = parser.parse_args()

    system_prompt, base_messages = _get_system_prompt_and_base_messages()

    if args.batch:
        plans_dir = Path(args.plans_dir) if args.plans_dir else Path(__file__).resolve().parent / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        tasks = [(c, p) for c in range(1000) for p in range(args.plans_per_class)]
        ok = skip = fail = 0
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, total: x
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _generate_one_plan,
                    c, p,
                    imagenet_idx2classname,
                    base_messages,
                    plans_dir,
                    args.max_retries,
                    args.skip_existing,
                ): (c, p)
                for c, p in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="plans"):
                status, cid, pid = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    fail += 1
        print(f"Batch done: ok={ok}, skip={skip}, fail={fail}", file=sys.stderr)
        sys.exit(0 if fail == 0 else 1)

    if args.class_id is None:
        raise SystemExit("Either --batch or --class-id required")
    class_id = args.class_id
    if class_id not in imagenet_idx2classname:
        raise SystemExit(f"Invalid --class-id {class_id}; must be 0-999.")
    category = imagenet_idx2classname[class_id]
    print(f"class_id: {class_id}, category: {category}", file=sys.stderr)

    messages = base_messages + [{"role": "user", "content": f"Category: {category}"}]
    for attempt in range(args.max_retries):
        reply = chat(
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        result = parse_matrix_response(reply)
        if result is not None:
            raster_indices = matrix_8x8_to_16x16_raster_indices(result)
            visualize_8x8(result)
            visualize_16x16_indices(raster_indices)
            plan = build_plan_dict(raster_indices, class_id=class_id, class_name=category)
            if args.output:
                with open(args.output, "wb") as f:
                    pickle.dump(plan, f)
                print(f"Saved to {args.output}", file=sys.stderr)
            print(plan)
            break
        if attempt < args.max_retries - 1:
            print(f"Invalid format (attempt {attempt + 1}/{args.max_retries}), retrying...", file=sys.stderr)
    else:
        raise SystemExit(f"Failed to get valid 8×8 matrix after {args.max_retries} attempts. Last reply:\n{reply}")
