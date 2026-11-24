from sentence_transformers import CrossEncoder

model = CrossEncoder("Qwen/Qwen3-Reranker-0.6B")
query = "广州今天的天气怎么样"
documents = [
    "下雨打车出门",
    "广州正在下雨",
    "广州美食不错",
    "达州正在下雨",
]
pairs = [(query, document) for document in documents]
scores = model.predict(pairs)
print(scores)
rankings = model.rank(query, documents)
print(rankings)
