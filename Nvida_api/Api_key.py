from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

completion = client.chat.completions.create(
    model="qwen/qwen3-next-80b-a3b-instruct",
    messages=[
        {"role": "user", "content": "你好."}
    ],
    temperature=0.6,
    max_tokens=512
)

print(completion.choices[0].message.content)