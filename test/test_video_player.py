import requests

from urllib import parse


def video_play(
    url: str,
    type: str,
    codec: str,
) -> int:
    size = 0
    with requests.get(
        f"http://localhost:8888/v1/video/play?url={url}&type={type}&codec={codec}",
        auth=("admin", "admin"),
        stream=True,
        timeout=(5, 60),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            size += len(line)
            if type == "device":
                # 摄像头要直接退出
                break
    return size


# fmt: off
urls = [
    ("./caiwei.mp4", "file", "true"),
    ("./caiwei.mp4", "file", "true"),
    ("./caiwei.mp4", "file", "false"),
    ("./caiwei.mp4", "file", "false"),
    (parse.quote('audio=麦克风阵列 (适用于数字麦克风的英特尔® 智音技术):video=Integrated Camera'), "device", "true"),
]
# fmt: on
index = 0
while True:
    if index >= 1000:
        break
    size = video_play(*urls[index % len(urls)])
    assert size > 0
    print(f"{index} => {size}")
    index += 1
