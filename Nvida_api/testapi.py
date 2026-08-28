import os
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("请先设置环境变量 OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

try:
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "你好，请回复：OpenRouter API 测试成功"
            }
        ],
    )

    print("API 调用成功！")
    print("模型：", response.model)
    print("回复：", response.choices[0].message.content)

except Exception as e:
    print("API 调用失败：")
    print(e)