#include "caiwei/log.hpp"
#include "caiwei/player.hpp"

#include "SDL2/SDL.h"

static int audio_sample_rate = 8000;
static int audio_nb_channels = 1;

struct PlayerState {
    bool running       = false;
    bool audio_running = false;
    bool video_running = false;
    int  video_width   = 0;
    int  video_height  = 0;
    SDL_mutex   *     mutex      = nullptr;
    SDL_Window  *     window     = nullptr;
    SDL_Renderer*     renderer   = nullptr;
    SDL_Texture *     texture    = nullptr;
    SDL_GLContext     context    = nullptr;
    SDL_AudioDeviceID audio_id   = 0;
    SDL_AudioSpec     audio_spec = {};
};

static PlayerState player_state = {};

static bool init_audio_player();
static bool init_video_player();
static void stop_audio_player();
static void stop_video_player();

bool caiwei::player::open_player(int channel, int sample_rate, int video_width, int video_height) {
    player_state.audio_spec = {
        .freq     = sample_rate,
        .format   = AUDIO_S16,
        .channels = static_cast<uint8_t>(channel),
        .silence  = 0,
        .samples  = 4800,
        .padding  = 0,
        .size     = 9600,
        .callback = nullptr,
        .userdata = nullptr
    };
    player_state.video_width  = video_width;
    player_state.video_height = video_height;
    int ret = SDL_Init(SDL_INIT_AUDIO | SDL_INIT_VIDEO);
    if(ret != 0) {
        LOG_WARN("加载播放器失败: %s", SDL_GetError());
        return false;
    }
    if(init_audio_player() && init_video_player()) {
        SDL_Event event;
        LOG_INFO("打开播放器成功");
        player_state.running = true;
        while(player_state.running) {
            SDL_WaitEventTimeout(&event, 1000);
            if(event.type == SDL_QUIT) {
                LOG_INFO("退出播放器");
                break;
            } else {
                // -
            }
        }
    } else {
        LOG_WARN("打开播放器失败");
    }
    stop_audio_player();
    stop_video_player();
    SDL_Quit();
    return true;
}

void caiwei::player::stop_player() {
    Uint32 flags = SDL_INIT_AUDIO | SDL_INIT_VIDEO;
    if(SDL_WasInit(flags) == flags) {
        SDL_Event event;
        event.type = SDL_QUIT;
        int ret = SDL_PushEvent(&event);
        LOG_INFO("关闭播放器: %d", ret);
    }
}

bool caiwei::player::play_audio(const void* data, int len) {
    if(player_state.running && player_state.audio_running) {
        int ret = SDL_QueueAudio(player_state.audio_id, data, len);
        if(ret != 0) {
            LOG_WARN("音频播放失败: %s", SDL_GetError());
            return false;
        }
        return true;
    }
    return false;
}

bool caiwei::player::play_video(const void* data, int len) {
    if(player_state.running && player_state.video_running) {
        int ret = SDL_LockMutex(player_state.mutex);
        if(ret != 0) {
            LOG_WARN("视频加锁失败: %s", SDL_GetError());
            return false;
        }
        ret = SDL_GL_MakeCurrent(SDL_GL_GetCurrentWindow(), SDL_GL_GetCurrentContext());
        if(ret != 0) {
            LOG_WARN("窗口绑定失败: %s", SDL_GetError());
            return false;
        }
        ret = SDL_UpdateTexture(player_state.texture, nullptr, data, len);
        if(ret != 0) {
            LOG_WARN("视频更新失败: %s", SDL_GetError());
            return false;
        }
        ret = SDL_RenderClear(player_state.renderer);
        if(ret != 0) {
            LOG_WARN("视频清除失败: %s", SDL_GetError());
            return false;
        }
        ret = SDL_RenderCopy(player_state.renderer, player_state.texture, nullptr, nullptr);
        if(ret != 0) {
            LOG_WARN("视频拷贝失败: %s", SDL_GetError());
            return false;
        }
        SDL_RenderPresent(player_state.renderer);
        ret = SDL_UnlockMutex(player_state.mutex);
        if(ret != 0) {
            LOG_WARN("视频解锁失败: %s", SDL_GetError());
            return false;
        }
        return true;
    }
    return false;
}

static bool init_audio_player() {
    if(player_state.audio_running) {
        LOG_INFO("音频已经打开");
        return true;
    }
    player_state.audio_id = SDL_OpenAudioDevice(nullptr, 0, &player_state.audio_spec, nullptr, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
    if(player_state.audio_id == 0) {
        LOG_WARN("打开音频失败: %s", SDL_GetError());
        return false;
    }
    SDL_PauseAudioDevice(player_state.audio_id, 0);
    player_state.audio_running = true;
    return true;
}

static bool init_video_player() {
    if(player_state.video_running) {
        LOG_INFO("视频已经打开");
        return true;
    }
    player_state.mutex = SDL_CreateMutex();
    if(!player_state.mutex) {
        LOG_WARN("打开互斥失败: %s", SDL_GetError());
        return false;
    }
    player_state.window = SDL_CreateWindow("Caiwei Player", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, player_state.video_width, player_state.video_height, SDL_WINDOW_OPENGL);
    if(!player_state.window) {
        LOG_WARN("打开窗口失败: %s", SDL_GetError());
        return false;
    }
    player_state.renderer = SDL_CreateRenderer(player_state.window, -1, 0);
    if(!player_state.renderer) {
        LOG_WARN("打开渲染失败: %s", SDL_GetError());
        return false;
    }
    player_state.texture = SDL_CreateTexture(player_state.renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, player_state.video_width, player_state.video_height);
    if(!player_state.texture) {
        LOG_WARN("打开纹理失败: %s", SDL_GetError());
        return false;
    }
    player_state.context = SDL_GL_CreateContext(player_state.window);
    if(!player_state.context) {
        LOG_WARN("打开OpenGL失败: %s", SDL_GetError());
        return false;
    }
    player_state.video_running = true;
    return true;
}

static void stop_audio_player() {
    LOG_INFO("关闭音频播放器");
    player_state.audio_running = false;
    if(player_state.audio_id != 0) {
        SDL_CloseAudioDevice(player_state.audio_id);
        player_state.audio_id = 0;
    }
}

static void stop_video_player() {
    LOG_INFO("关闭视频播放器");
    player_state.video_running = false;
    if(player_state.context) {
        SDL_GL_DeleteContext(player_state.context);
        player_state.context = nullptr;
    }
    if(player_state.texture) {
        SDL_DestroyTexture(player_state.texture);
        player_state.texture = nullptr;
    }
    if(player_state.renderer) {
        SDL_DestroyRenderer(player_state.renderer);
        player_state.renderer = nullptr;
    }
    if(player_state.window) {
        SDL_DestroyWindow(player_state.window);
        player_state.window = nullptr;
    }
    if(player_state.mutex) {
        SDL_DestroyMutex(player_state.mutex);
        player_state.mutex = nullptr;
    }
}
