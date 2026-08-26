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
    port: 9999,
    host: "0.0.0.0",
  },
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
});
