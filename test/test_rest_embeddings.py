from openai import OpenAI

client = OpenAI(api_key="acgist", base_url="http://127.0.0.1:8888/v1")
resp = client.embeddings.create(
    model="Qwen3-Embedding-0.6B",
    input=["碧螺萧萧"],
)
print(resp)
