import requests


def video_play():
    resp = requests.get(
        "http://localhost:8888/v1/video/play?url=./caiwei.mp4&type=file&codec=true",
        stream=True,
    )
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        print(len(line))


index = 0
while True:
    index += 1
    if index > 100:
        break
    video_play()
