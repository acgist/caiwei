<template>
  <div>
    <el-row>
      <div class="tool-bar" style="margin:auto;">
        <el-select v-model="type" style="width:8rem;margin-right:1rem;">
          <el-option value="AudioASR"    label="语音识别"></el-option>
          <el-option value="AudioTTS"    label="语音合成"></el-option>
          <el-option value="ImageYolo"   label="图片识别"></el-option>
          <el-option value="VideoPlay"   label="媒体播放"></el-option>
          <el-option value="VideoYolo"   label="视频识别"></el-option>
          <el-option value="Rerankings"  label="重新排序"></el-option>
          <el-option value="Embeddings"  label="嵌入向量"></el-option>
          <el-option value="Completions" label="文本生成"></el-option>
        </el-select>
        <el-input style="width:16rem;" v-model="host">
          <template #suffix>
            <el-icon :color="status"><MostlyCloudy /></el-icon>
          </template>
        </el-input>
      </div>
    </el-row>
    <audio-asr   v-if="type === 'AudioASR'   " :host="host" ref="audioASR"   ></audio-asr>
    <audio-tts   v-if="type === 'AudioTTS'   " :host="host" ref="audioTTS"   ></audio-tts>
    <image-yolo  v-if="type === 'ImageYolo'  " :host="host" ref="imageYolo"  ></image-yolo>
    <video-play  v-if="type === 'VideoPlay'  " :host="host" ref="videoPlay"  ></video-play>
    <video-yolo  v-if="type === 'VideoYolo'  " :host="host" ref="videoYolo"  ></video-yolo>
    <rerankings  v-if="type === 'Rerankings' " :host="host" ref="rerankings" ></rerankings>
    <embeddings  v-if="type === 'Embeddings' " :host="host" ref="embeddings" ></embeddings>
    <completions v-if="type === 'Completions'" :host="host" ref="completions"></completions>
  </div>
</template>

<script>
import AudioAsr from "./AudioASR.vue";
import AudioTts from "./AudioTTS.vue";
import Completions from "./Completions.vue";
import Embeddings from "./Embeddings.vue";
import ImageYolo from "./ImageYolo.vue";
import Rerankings from "./Rerankings.vue";
import VideoPlay from "./VideoPlay.vue";
import VideoYolo from "./VideoYolo.vue";
export default {
  
  name: "App",
  components: {
    AudioAsr,
    AudioTts,
    ImageYolo,
    VideoPlay,
    VideoYolo,
    Rerankings,
    Embeddings,
    Completions,
  },
  data() {
    return {
      host  : `http://${location.hostname}:8888`,
      type  : "VideoPlay",
      status: "green",
    }
  },
  mounted() {
    // setInterval(async () => {
    //   try {
    //     const resp = await fetch(`${this.host}/v1/health`);
    //     const json = await resp.json();
    //     if(json.header.code === "0000") {
    //       this.status = "green";
    //     } else {
    //       this.status = "red";
    //     }
    //   } catch (error) {
    //     this.status = "red";
    //   }
    // }, 5000);
  },
};
</script>
