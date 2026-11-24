from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
queries = [
    "小狗",
]
documents = [
    "猫咪",
    "苹果",
    "汽车",
]
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
