from sentence_transformers import CrossEncoder

model = CrossEncoder("Qwen/Qwen3-Reranker-0.6B")
query = "中国的首都是是哪个城市？"
documents = [
    "北京",
    "中国的首都是北京",
    "广州是一座美丽的城市",
    "成都是四川的省会城市",
]
pairs = [(query, document) for document in documents]
scores = model.predict(pairs)
print(scores)
rankings = model.rank(query, documents)
print(rankings)
