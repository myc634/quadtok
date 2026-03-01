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
You are given a single object category. 
Your task is to generate a typical spatial complexity map of an assumed ImageNet style picture of the object. 
Divide the image into an 8 * 8 grid. 
Output a binary 8 * 8 matrix where: 
1 = this region is typically occupied by the main object.
0 = the rest of the part where it should be easy to describe the details

Guidance: 
Be creative about the object orientation, placement, position and size. 
A good matrix is a little noisy, no LARGE clusters, but with several isolated 1s and several isolated 0s.

Output format: 
First simply describe the assumed image. 
Then generate an 8 * 8 matrix of 0s(shown as 🟦) and 1s(shown as 🟥) based on your description. 
Each row on a new line Exactly 8 numbers per row, no space between them.
    """

    # prompt = """
    #         You are given a single object category. 
    #         Your task is to generate a typical spatial complexity map of an assumed ImageNet style picture of the object. 
    #         Divide the image into an 8 × 8 grid. 
    #         Output a binary 8 × 8 matrix where: 
    #             1 = this region is typically occupied by the main object or area which high contrast, high frequency information and theoretically need more level of detail to describe 
    #             0 = the rest of the part where it should be easy to describe the details 
            
    #         Important clarifications:
    #             1s do NOT necessarily correspond only to the main object but should reflect the typical structural complexity patterns for this category.
    #             Background regions may also be marked as 1 if they are typically visually complex.
    #             The 1s should NOT be completely contiguous.
    #             Avoid simply marking a centered block by default.
    #             The activation regions ratio should not be larger than 0.6.

    #         First simply describe the assumed image. 
    #         Then generate an 8 × 8 matrix of 0s and 1s based on your description. 
    #         Each row on a new line Exactly 8 numbers per row.

    #         Output format: 
    #         Response ONLY the description of the image and the 8 × 8 matrix of 0s (shown as 🟦) and 1s (shown as 🟥), no space between them.           
    # """

    reply = chat(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Categoey: {args.category}"},
        ],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    print("Reply:", reply)
