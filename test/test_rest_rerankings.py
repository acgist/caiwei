import json
import requests

url = "http://127.0.0.1:8888/v1/rerankings"
payload = {
    "model": "qwen3-reranking",
    "query": "广州今天的天气怎么样",
    "documents": [
        "下雨打车出门",
        "广州正在下雨",
        "广州美食不错",
        "达州正在下雨",
    ],
}
headers = {"Content-Type": "application/json"}
resp = requests.post(url, data=json.dumps(payload), headers=headers)
resp.raise_for_status()
data = resp.json()
print(data["usage"])
for reranking in data["data"]:
    print(f"{reranking['index']} -> {reranking['score']:.4f}")
