from openai import OpenAI

client = OpenAI(api_key="acgist", base_url="http://127.0.0.1:8888/v1")
resp = client.embeddings.create(
    model="qwen3-embedding",
    input=["碧螺萧萧", "广州", "北京"],
)
print(resp.usage)
for embedding in resp.data:
    print(f"{embedding.index} -> len = {len(embedding.embedding)}")
