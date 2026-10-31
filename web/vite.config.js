import vue from "@vitejs/plugin-vue";
import {
  URL,
  fileURLToPath,
} from "url";
import {
  defineConfig
} from "vite";

export default defineConfig({
  plugins: [vue()],
  server: {
    host: "0.0.0.0",
    port: 9999,
  },
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
});
