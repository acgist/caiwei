#include "player.hpp"

#include "SDL2/SDL.h"

#include <mutex>
#include <thread>
#include <condition_variable>

#include "caiwei/log.hpp"

const static int USER_EVENT_VIDEO_FRAME = 1;

struct PlayerState {
    bool init_done     = false;
    bool running       = false;
    bool audio_running = false;
    bool video_running = false;
    bool video_pending = false;
    int  video_width   = 0;
    int  video_height  = 0;
    std::mutex player_mutex;
    std::thread player_thread;
    std::condition_variable player_cv;
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
    player_state.init_done = false;
    player_state.audio_spec = {
        .freq     = sample_rate,
        .format   = AUDIO_S16,
        .channels = static_cast<uint8_t>(channel),
        .silence  = 0,
        .samples  = 4800,
        .padding  = 0,
        .size     = 9600,
        .callback = nullptr,
        .userdata = nullptr,
    };
    player_state.video_width  = video_width;
    player_state.video_height = video_height;
    if (player_state.player_thread.joinable()) {
        player_state.player_thread.join();
    }
    player_state.player_thread = std::thread([]() {
        int ret = SDL_Init(SDL_INIT_AUDIO | SDL_INIT_VIDEO);
        if(ret != 0) {
            CW_LOG_W("加载播放器失败: %s", SDL_GetError());
            return;
        }
        if(init_audio_player() && init_video_player()) {
            CW_LOG_I("打开播放器成功");
            {
                std::lock_guard<std::mutex> lock(player_state.player_mutex);
                player_state.init_done = true;
                player_state.player_cv.notify_one();
            }
            SDL_Event event;
            player_state.running = true;
            while(player_state.running) {
                int ret_event = SDL_WaitEventTimeout(&event, 1000);
                if (ret_event == 0) {
                    continue;
                }
                if(event.type == SDL_QUIT) {
                    CW_LOG_I("退出播放器");
                    break;
                } else if (event.type == SDL_KEYUP) {
                    if (event.key.keysym.sym == SDLK_q) {
                        break;
                    }
                } else if (event.type == SDL_USEREVENT) {
                    if (event.user.code == USER_EVENT_VIDEO_FRAME) {
                        if(SDL_LockMutex(player_state.mutex) != 0) {
                            CW_LOG_W("视频加锁失败: %s", SDL_GetError());
                        } else {
                            if (SDL_RenderClear(player_state.renderer) != 0) {
                                CW_LOG_W("视频清除失败: %s", SDL_GetError());
                            } else {
                                if (SDL_RenderCopy(player_state.renderer, player_state.texture, nullptr, nullptr) != 0) {
                                    CW_LOG_W("视频拷贝失败: %s", SDL_GetError());
                                } else {
                                    SDL_RenderPresent(player_state.renderer);
                                }
                            }
                            player_state.video_pending = false;
                            SDL_UnlockMutex(player_state.mutex);
                        }
                    } else {
                        // -
                    }
                } else {
                    // -
                }
            }
        } else {
            CW_LOG_W("打开播放器失败");
            {
                std::lock_guard<std::mutex> lock(player_state.player_mutex);
                player_state.init_done = true;
                player_state.player_cv.notify_one();
            }
        }
        stop_audio_player();
        stop_video_player();
        SDL_Quit();
    });
    std::unique_lock<std::mutex> lock(player_state.player_mutex);
    player_state.player_cv.wait_for(lock, std::chrono::seconds(10), []() {
        return player_state.init_done;
    });
    return player_state.running;
}

void caiwei::player::stop_player() {
    Uint32 flags = SDL_INIT_AUDIO | SDL_INIT_VIDEO;
    if(SDL_WasInit(flags) == flags) {
        SDL_Event event;
        event.type = SDL_QUIT;
        int ret = SDL_PushEvent(&event);
        CW_LOG_I("关闭播放器: %d", ret);
    }
    if (player_state.player_thread.joinable()) {
        player_state.player_thread.join();
    }
}

bool caiwei::player::play_audio(const void* data, int len) {
    if(player_state.running && player_state.audio_running) {
        int ret = SDL_QueueAudio(player_state.audio_id, data, len);
        if(ret != 0) {
            CW_LOG_W("音频播放失败: %s", SDL_GetError());
            return false;
        }
        return true;
    }
    return false;
}

bool caiwei::player::play_video(const void* data, int len) {
    if(player_state.running && player_state.video_running) {
        if(SDL_LockMutex(player_state.mutex) != 0) {
            CW_LOG_W("视频加锁失败: %s", SDL_GetError());
        } else {
            bool push = false;
            if(SDL_UpdateTexture(player_state.texture, nullptr, data, len) != 0) {
                CW_LOG_W("视频更新失败: %s", SDL_GetError());
            } else {
                if (!player_state.video_pending) {
                    push = true;
                    player_state.video_pending = true;
                }
            }
            SDL_UnlockMutex(player_state.mutex);
            if (push) {
                SDL_Event event;
                event.type = SDL_USEREVENT;
                event.user.code = USER_EVENT_VIDEO_FRAME;
                SDL_PushEvent(&event);
            }
        }
        return true;
    }
    return false;
}

static bool init_audio_player() {
    if(player_state.audio_running) {
        CW_LOG_I("音频已经打开");
        return true;
    }
    player_state.audio_id = SDL_OpenAudioDevice(nullptr, 0, &player_state.audio_spec, nullptr, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
    if(player_state.audio_id == 0) {
        CW_LOG_W("打开音频失败: %s", SDL_GetError());
        return false;
    }
    SDL_PauseAudioDevice(player_state.audio_id, 0);
    player_state.audio_running = true;
    return true;
}

static bool init_video_player() {
    if(player_state.video_running) {
        CW_LOG_I("视频已经打开");
        return true;
    }
    player_state.mutex = SDL_CreateMutex();
    if(!player_state.mutex) {
        CW_LOG_W("打开互斥失败: %s", SDL_GetError());
        return false;
    }
    player_state.window = SDL_CreateWindow("Caiwei Player", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, player_state.video_width, player_state.video_height, SDL_WINDOW_OPENGL);
    if(!player_state.window) {
        CW_LOG_W("打开窗口失败: %s", SDL_GetError());
        return false;
    }
    player_state.renderer = SDL_CreateRenderer(player_state.window, -1, 0);
    if(!player_state.renderer) {
        CW_LOG_W("打开渲染失败: %s", SDL_GetError());
        return false;
    }
    player_state.texture = SDL_CreateTexture(player_state.renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, player_state.video_width, player_state.video_height);
    if(!player_state.texture) {
        CW_LOG_W("打开纹理失败: %s", SDL_GetError());
        return false;
    }
    player_state.context = SDL_GL_CreateContext(player_state.window);
    if(!player_state.context) {
        CW_LOG_W("打开OpenGL失败: %s", SDL_GetError());
        return false;
    }
    player_state.video_running = true;
    return true;
}

static void stop_audio_player() {
    CW_LOG_I("关闭音频播放器");
    player_state.audio_running = false;
    if(player_state.audio_id != 0) {
        SDL_CloseAudioDevice(player_state.audio_id);
        player_state.audio_id = 0;
    }
}

static void stop_video_player() {
    CW_LOG_I("关闭视频播放器");
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
