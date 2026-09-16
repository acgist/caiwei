import json
import requests

url = "http://127.0.0.1:8888/v1/rerankings"
payload = {
    "model": "Qwen3-Reranker-0.6B",
    "query": "中国首都是哪里",
    "documents": [
        "北京",
        "中国首都是北京",
        "四川是个美丽的城市",
    ],
}
headers = {"Content-Type": "application/json"}
resp = requests.post(url, data=json.dumps(payload), headers=headers)
print(resp.status_code)
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
