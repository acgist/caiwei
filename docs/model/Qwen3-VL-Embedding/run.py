from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

queries = [
    {"text": "人骑着马还有汽车。"},
    {
        "text": "四个女生分别穿着淡蓝色长袖防晒罩衫、白色长袖衬衣、灰色运动内衣和浅紫色吊带。"
    },
]
documents = [
    {"text": "四个穿着不同的女生。"},
    {"image": "/data/caiwei/test/acgist.jpg"},
    {
        "video": "/data/caiwei/test/caiwei.mp4",
        "fps": 2,
        "max_frames": 8,
    },
    {
        "text": "四个女生分别穿着白色短裤、蓝色牛仔裤和白色短裙。",
        "image": "/data/caiwei/test/acgist.jpg",
    },
]
model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")
inputs = queries + documents
embeddings = model.process(inputs)
similarity_scores = embeddings[:2] @ embeddings[2:].T
print(similarity_scores.tolist())

"""
export PYTHONPATH="Qwen/Qwen3-VL-Embedding-2B"
"""
