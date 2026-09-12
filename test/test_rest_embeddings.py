from openai import OpenAI

client = OpenAI(api_key="acgist", base_url="http://192.168.1.100:8000/v1")
resp = client.embeddings.create(
    model="Qwen3-Embedding-0.6B",
    input=["碧螺萧萧"],
)
print(resp)
