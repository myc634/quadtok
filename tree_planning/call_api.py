"""
简单调用本地 vLLM (Qwen) API 的示例。
需要先启动: vllm serve Qwen/Qwen3.5-27B --port 8000 ...
"""
from openai import OpenAI

# vLLM 提供 OpenAI 兼容接口，base_url 指向本地服务
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # 本地 vLLM 不需要 key
)

# 模型名与 serve 时一致，通常为 "Qwen/Qwen3.5-27B" 或 vLLM 显示的 name
MODEL = "Qwen/Qwen3.5-27B"


def chat(messages: list[dict], **kwargs) -> str:
    """发一轮对话请求，返回助手回复内容。"""
    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        **kwargs,
    )
    return r.choices[0].message.content
# 1 = this region is typically occupied by the main object or area which high contrast, high frequency information and theoretically need more level of detail to describe 
# 0 = the rest of the part where it should be easy to describe the details 
# Guidance: 
# It should be clarified that 1s don't always mean the main object, the background part can also be 1 if it's potentially informative. In other words, isolated 1s and non-continental 1s are also welcomed. 
# Be creative about the object orientation, placement, position and size. 

            # 1 = this region is likely to contain visually complex content
            # (e.g., object structure, edges, texture, high contrast, structural variation, detailed background)
            # 0 = this region is likely to contain visually simple or smooth content
            # (e.g., clear sky, plain wall, blurred background, uniform areas)

            # Important clarifications:
            #     1s do NOT necessarily correspond only to the main object.
            #     Background regions may also be marked as 1 if they are typically visually complex.
            #     The map does NOT need to be contiguous.
            #     Avoid simply marking a centered block by default.
            #     However, assume a realistic canonical framing — not an extreme or unusual pose.
            #     The activated regions should reflect typical structural complexity patterns for this category.

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    args = parser.parse_args()

    prompt = """
You are given a single object category. Your task is to generate a typical **spatial complexity map** for an assumed ImageNet-style image of that category.

Think of the map as a **blocky, pixelated mask** (8×8 grid, like red squares on black): the pattern of 1s should **roughly suggest the object's shape** — so that looking at the red blocks, one can tell where the main object is (e.g. for a building, the 1s might form a dome or tower silhouette; for an object, its outline or main mass).

**Semantics:**
- **1 (🟥)** = this cell is on or near the **main object** (or its visually complex parts: structure, edges, texture). Prefer marking object region; only mark background as 1 if it is clearly complex (e.g. detailed foreground) and fits the composition.
- **0 (🟦)** = background, sky, plain wall, blur, or other visually simple areas.

**Shape of the map:**
- 1s should form a **readable object silhouette** at 8×8 resolution: connected regions (blobs, bands, L/T-shapes) that approximate where the object sits in the frame.
- **Internal voids are fine**: 0s inside the object region (e.g. sky visible through a dome, or simple patches on the object) make the map more natural; edges can be jagged/blocky.
- Avoid a single solid rectangle; the outline should be **blocky but recognizable** as the category. Be creative about object position and framing; activation ratio of 1s should not exceed 0.6.

**Output format:**
1. In 1–3 sentences, describe the assumed image (framing, object, background).
2. Output an 8×8 matrix: 🟥 for 1, 🟦 for 0. Exactly 8 symbols per row, one row per line, no spaces between symbols.
    """


    reply = chat(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Category: {args.category}"},
        ],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    print("Reply:", reply)
