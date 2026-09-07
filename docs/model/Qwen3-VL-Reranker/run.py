from scripts.qwen3_vl_reranker import Qwen3VLReranker

model = Qwen3VLReranker(model_name_or_path="Qwen/Qwen3-VL-Reranker-2B")
inputs = {
    "instruction": "Retrieve images or text relevant to the user's query.",
    "query": {
        "text": "四个女生分别穿着淡蓝色长袖防晒罩衫、白色长袖衬衣、灰色运动内衣和浅紫色吊带。"
    },
    "documents": [
        {"text": "四个穿着不同的女生。"},
        {"image": "/data/caiwei/test/acgist.jpg"},
        {"video": "/data/caiwei/test/caiwei.mp4"},
        {
            "text": "四个女生分别穿着白色短裤、蓝色牛仔裤和白色短裙。",
            "image": "/data/caiwei/test/acgist.jpg",
        },
    ],
    "fps": 2.0,
    "max_frames": 8,
}
scores = model.process(inputs)
print(scores)

"""
export PYTHONPATH="Qwen/Qwen3-VL-Reranker-2B"
"""
