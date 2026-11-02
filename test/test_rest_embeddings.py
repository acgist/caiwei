import numpy as np

from openai import OpenAI


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


client = OpenAI(api_key="caiwei", base_url="http://127.0.0.1:8888/v1")
resp = client.embeddings.create(
    model="qwen3-embedding",
    input=[
        "小狗",
        "猫咪",
        "苹果",
        "汽车",
    ],
)
print(resp.usage)
for embedding in resp.data:
    print(f"{embedding.index} -> len = {len(embedding.embedding)}")
print(f"{cosine_similarity(resp.data[0].embedding, resp.data[1].embedding):.4f}")
print(f"{cosine_similarity(resp.data[0].embedding, resp.data[2].embedding):.4f}")
print(f"{cosine_similarity(resp.data[0].embedding, resp.data[3].embedding):.4f}")
print(f"{cosine_similarity(resp.data[1].embedding, resp.data[2].embedding):.4f}")
print(f"{cosine_similarity(resp.data[1].embedding, resp.data[3].embedding):.4f}")
print(f"{cosine_similarity(resp.data[2].embedding, resp.data[3].embedding):.4f}")
