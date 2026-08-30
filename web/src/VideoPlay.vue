<template>
  <div style="display:flex;align-items:center;justify-content:center;padding:1rem;">
    <video ref="localVideo" style="width:640px;height:320px;background:#f0f0f0;" controls></video>
  </div>
  <div style="display:flex;align-items:center;justify-content:center;">
    <el-button @click="openMedia" type="success">播放媒体</el-button>
    <el-button @click="stopMedia" type="danger" >关闭媒体</el-button>
  </div>
  <div style="display:flex;align-items:center;justify-content:center;padding:1rem;">
    <el-select v-model="type" style="width:6rem;margin-right:1rem;" @change="changeType">
          <el-option value="rtp"    label="rtp"   ></el-option>
          <el-option value="sdp"    label="sdp"   ></el-option>
          <el-option value="file"   label="file"  ></el-option>
          <el-option value="http"   label="http"  ></el-option>
          <el-option value="rtmp"   label="rtmp"  ></el-option>
          <el-option value="rtsp"   label="rtsp"  ></el-option>
          <el-option value="device" label="device"></el-option>
        </el-select>
    <el-input style="width:16rem;" v-model="url" placeholder="媒体地址"></el-input>
  </div>
</template>

<script>
export default {
  name: "VideoPlay",
  props: {
    host: {
      type   : String,
      default: `http://${location.hostname}:8888`,
    },
  },
  data() {
    return {
      type  : "file",
      url   : null,
      urls  : {
        // ffmpeg -re -i caiwei.mp4 -c:a pcm_alaw -ar 8000 -ac 1 -vn -f rtp -payload_type  8 -ssrc 1000 rtp://127.0.0.1:44444 > audio.sdp
        // ffmpeg -re -i caiwei.mp4 -c:v libx264                 -an -f rtp -payload_type 96 -ssrc 2000 rtp://127.0.0.1:44446 > video.sdp
        rtp   : `v=0
c=IN IP4 0.0.0.0
o=- 0 0 IN IP4 0.0.0.0
s=caiwei rtp capture
t=0 0
m=audio 44444 RTP/AVP 8
a=rtcp:44445
a=rtpmap:8 PCMA/8000/1
a=recvonly
m=video 44446 RTP/AVP 96
a=rtcp:44447
a=rtpmap:96 H264/90000
a=fmtp:96 profile-level-id=42e01f;packetization-mode=1;level-asymmetry-allowed=1
a=recvonly`,
        sdp   : `v=0
c=IN IP4 0.0.0.0
o=- 0 0 IN IP4 0.0.0.0
s=caiwei rtp capture
t=0 0
m=audio 44444 RTP/AVP 8
a=rtcp:44445
a=rtpmap:8 PCMA/8000/1
a=recvonly
m=video 44446 RTP/AVP 96
a=rtcp:44447
a=rtpmap:96 H264/90000
a=fmtp:96 profile-level-id=42e01f;packetization-mode=1;level-asymmetry-allowed=1
a=recvonly`,
        file  : "./caiwei.mp4",
        http  : "http://localhost:9999/caiwei.mp4",
        rtmp  : "rtmp://liteavapp.qcloud.com/live/liteavdemoplayerstreamid",
        rtsp  : "rtsp://admin:admin@127.0.0.1:554/h264/ch1/main/av_stream",
        device: "audio=麦克风阵列 (适用于数字麦克风的英特尔® 智音技术):video=Integrated Camera",
      },
      queue : [],
      media : null,
      event : null,
      buffer: null,
    };
  },
  mounted() {
    this.changeType();
  },
  unmounted() {
    this.stopMedia();
  },
  methods: {
    async changeType() {
      this.url = this.urls[this.type];
    },
    async openMedia() {
      if(this.event && this.media) {
        this.$message.error("媒体已经打开");
        return;
      }
      this.event = new EventSource(`${this.host}/v1/video/play?type=${this.type}&url=${encodeURIComponent(this.url)}`);
      this.event.onopen = async e => {
        this.media = new MediaSource();
        const localVideo = this.$refs.localVideo;
        localVideo.src = URL.createObjectURL(this.media);
        this.media.addEventListener("sourceopen", async () => {
          const mime = 'video/mp4; codecs="avc1.42C020, mp4a.40.2"';
          this.buffer = this.media.addSourceBuffer(mime);
          this.buffer.addEventListener("updateend", () => {
            if (this.buffer.updating) {
              return;
            }
            const buffered = this.buffer.buffered;
            if (buffered.length === 0) {
              return;
            }
            const playTime = localVideo.currentTime;
            const buffTime = this.buffer.buffered.start(0);
            // 删除当前播放时间30秒钟以前缓存数据
            if(playTime - buffTime > 60) {
              this.buffer.remove(0, playTime - 30);
            }
          });
          this.buffer.addEventListener("error", e => {
            console.error("媒体缓冲异常", e);
            this.stopMedia();
          });
          await localVideo.play();
        });
        this.media.addEventListener("sourceended", () => {
          console.info("媒体结束");
          this.stopMedia();
        });
        this.media.addEventListener("sourceclose", () => {
          console.info("媒体关闭");
          this.stopMedia();
        });
        this.media.addEventListener("error", e => {
          console.error("媒体异常", e);
          this.stopMedia();
        });
      };
      this.event.onerror = e => {
        console.error("事件异常", e);
        this.stopMedia();
      };
      this.event.addEventListener("data", e => {
        if(!this.buffer) {
          console.warn("媒体没有准备就绪");
          return;
        }
        const raw  = atob(e.data);
        const blob = new Uint8Array(raw.length);
        for(let index = 0; index < raw.length; ++index) {
          blob[index] = raw.charCodeAt(index);
        }
        this.queue.push(blob);
        if(this.buffer.updating) {
          // 更新中不处理
          return;
        }
        if(this.queue.length === 1) {
          this.buffer.appendBuffer(this.queue.shift());
          return;
        }
        let length = 0;
        let offset = 0;
        for (const value of this.queue) {
          length += value.byteLength;
        }
        const array = new Uint8Array(length);
        for (const value of this.queue) {
            const frame = new Uint8Array(value);
            array.set(frame, offset);
            offset += value.byteLength;
        }
        this.buffer.appendBuffer(array);
        this.queue.splice(0, this.queue.length);
      });
    },
    async stopMedia() {
      if(this.event && this.media) {
        console.info("关闭媒体");
        this.queue = [];
        this.event.close();
        this.event = null;
        this.media.removeSourceBuffer(this.buffer);
        this.media = null;
        this.buffer = null;
        const localVideo = this.$refs.localVideo;
        localVideo.pause();
        URL.revokeObjectURL(localVideo.src);
        localVideo.src = "";
        localVideo.load();
      } else {
        console.info("媒体已经关闭");
      }
    },
  },
};
</script>
