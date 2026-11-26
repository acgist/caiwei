<template>
  <div>
    <el-row>
      <div class="tool-bar" style="margin:auto;">
        <el-input style="width:16rem;" v-model="host">
          <template #suffix>
            <el-icon :color="status"><MostlyCloudy /></el-icon>
          </template>
        </el-input>
      </div>
    </el-row>
    <video-play :host="host" ref="videoPlay"></video-play>
  </div>
</template>

<script>
import VideoPlay from "./VideoPlay.vue";
export default {
  name: "App",
  components: {
    VideoPlay,
  },
  data() {
    return {
      host  : `http://${location.hostname}:8888`,
      status: "#00cc66",
      healthTimerId: 0,
    }
  },
  mounted() {
    this.healthTimer();
  },
  unmounted() {
    this.clearHealthTimer();
  },
  methods: {
    async healthCheck() {
      try {
          const resp = await fetch(`${this.host}/v1/health`);
          const json = await resp.json();
          if (json.header.code === "0000") {
            this.status = "#00cc66";
          } else {
            this.status = "#ff0000";
          }
        } catch (error) {
          this.status = "#ff0000";
        }
    },
    healthTimer() {
      if (this.healthTimerId != 0) {
        return;
      }
      this.healthCheck();
      this.healthTimerId = setInterval(async () => {
        await this.healthCheck();
      }, 5000);
    },
    clearHealthTimer() {
      clearInterval(this.healthTimerId);
      this.healthTimerId = 0;
    },
  },
};
</script>
