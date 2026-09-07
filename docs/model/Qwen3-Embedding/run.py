from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
queries = [
    "中国的首都是是哪个城市？",
    "想出去旅游有什么推荐的城市？",
]
documents = [
    "北京",
    "中国的首都是北京",
    "广州是一座美丽的城市，是一个热门的旅游目的地。",
    "成都是四川的省会城市，是一个热门的旅游目的地。",
]
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
