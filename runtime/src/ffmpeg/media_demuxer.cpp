#include <mutex>
#include <thread>
#include <fstream>
#include <filesystem>

#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/media_tool.hpp"

extern "C" {
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
#include "libswresample/swresample.h"
}

static bool write_sdp_file(const char* data, int size, const std::filesystem::path& path);

static SwrContext* init_audio_swr(caiwei::media::AudioInfo& audio_info, AVFrame* frame);
static SwsContext* init_video_sws(caiwei::media::VideoInfo& video_info, AVFrame* frame);

static void sws_free(SwsContext** sws);

static bool init_decoder_ctx(const std::string& type, const std::string& url, AVCodecContext** decoder_ctx, AVCodecParameters* codecpar);

static bool device_open = false;
static std::mutex device_mutex;

caiwei::media::MediaDemuxer::MediaDemuxer(
    const std::string& type,
    const std::string& url
) : running(false), type(type), url(url) {
}

inline bool recv_audio_frame(
    const size_t msec,
    const size_t audio_frames,
    SwrContext* swr_ctx,
    const AVFrame* frame,
    const caiwei::media::AudioInfo& audio_info,
    caiwei::media::AudioFrame& audio_frame
) {
    int nb_samples = swr_get_out_samples(swr_ctx, frame->nb_samples);
    if (nb_samples < 0) {
        CW_LOG_W("音频重采样失败: %d", nb_samples);
        return false;
    }
    audio_frame.data.resize(nb_samples * audio_info.channels * audio_info.bytes_per_sample);
    uint8_t* dst_data = audio_frame.data.data();
    nb_samples = swr_convert(swr_ctx, &dst_data, nb_samples, (const uint8_t**) frame->data, frame->nb_samples);
    if (nb_samples < 0) {
        CW_LOG_W("音频重采样失败: %d", nb_samples);
        return false;
    }
    audio_frame.msec    = msec;
    audio_frame.frames  = audio_frames;
    audio_frame.samples = nb_samples;
    audio_frame.data.resize(nb_samples * audio_info.channels * audio_info.bytes_per_sample);
    return true;
}

inline bool recv_video_frame(
    const size_t msec,
    const size_t video_frames,
    SwsContext* sws_ctx,
    const AVFrame* frame,
    const caiwei::media::VideoInfo& video_info,
    caiwei::media::VideoFrame& video_frame
) {
    const int dst_stride = video_info.width * video_frame.channels;
    video_frame.data.resize(video_info.width * video_info.height * video_frame.channels);
    uint8_t* dst_data = video_frame.data.data();
    const int dst_height = sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, &dst_data, &dst_stride);
    if (dst_height < 0) {
        CW_LOG_W("视频重采样失败: %d", dst_height);
        return false;
    }
    if (dst_height != video_info.height) {
        CW_LOG_W("视频重采样失败: %d != %d", dst_height, video_info.height);
        return false;
    }
    video_frame.width  = video_info.width;
    video_frame.height = dst_height;
    video_frame.msec   = msec;
    video_frame.frames = video_frames;
    return true;
}

bool caiwei::media::MediaDemuxer::open(
    AudioInfo audio_info,
    VideoInfo video_info,
    AudioCallback audio_callback,
    VideoCallback video_callback,
    MediaFormat* media_format
) {
    int ret = AVERROR_UNKNOWN;
    bool open_device = false;
    std::string sdp_file;
    int64_t msec = 0;
    size_t audio_frames  = 0;
    size_t video_frames  = 0;
    int audio_stream_idx = -1;
    int video_stream_idx = -1;
    int             audio_frame_format      = 0;
    AVChannelLayout audio_frame_ch_layout{};
    int             audio_frame_sample_rate = 0;
    int             video_frame_width       = 0;
    int             video_frame_height      = 0;
    int             video_frame_format      = 0;
    AVFrame * decoder_frame  = av_frame_alloc();
    AVPacket* decoder_packet = av_packet_alloc();
    AVDictionary   * opts    = nullptr;
    SwrContext     * swr_ctx = nullptr;
    SwsContext     * sws_ctx = nullptr;
    AVFormatContext* fmt_ctx = avformat_alloc_context();
    AVCodecContext * audio_decoder_ctx = nullptr;
    AVCodecContext * video_decoder_ctx = nullptr;
    const AVRational msec_time_base{ 1, 1000 };
    AVRational audio_time_base;
    AVRational video_time_base;
    auto timeout = caiwei::env::get_int("CAIWEI_TIMEOUT"); // 超时时间
    auto last_recv_time = std::chrono::steady_clock::now(); // 最后接收时间
    auto last_send_time = std::chrono::steady_clock::now(); // 最后发送时间
    caiwei::media::AudioFrame audio_frame;
    caiwei::media::VideoFrame video_frame;
    this->running = true;
    fmt_ctx->interrupt_callback.opaque   = this;
    fmt_ctx->interrupt_callback.callback = [](void* ptr) {
        auto demuxer = (caiwei::media::MediaDemuxer*) ptr;
        if (demuxer->running) {
            return 0;
        }
        CW_LOG_D("媒体中断回调");
        return AVERROR_EXIT;
    };
    av_dict_set(&opts, "seekable",          "0",                                               0);
    av_dict_set(&opts, "buffer_size",       "262144",                                          0);
    av_dict_set(&opts, "thread_queue_size", "2048",                                            0);
    av_dict_set(&opts, "protocol_whitelist", "tcp,tls,udp,rtp,data,file,http,https,rtmp,rtsp", 0);
    if (this->type == "rtp" || this->type == "sdp") {
        av_dict_set(&opts, "timeout",                     "10000000",               0);
        av_dict_set(&opts, "rw_timeout",                  "10000000",               0);
        av_dict_set(&opts, "fflags",                      "+genpts+discardcorrupt", 0);
        av_dict_set(&opts, "max_delay",                   "1000000",                0);
        av_dict_set(&opts, "reorder_queue_size",          "2048",                   0);
        av_dict_set(&opts, "use_wallclock_as_timestamps", "1",                      0);
        sdp_file = sdp_file_path + caiwei::env::id() + ".sdp";
        if (write_sdp_file(this->url.c_str(), this->url.size(), sdp_file)) {
            ret = avformat_open_input(&fmt_ctx, sdp_file.c_str(), nullptr, &opts);
        }
    } else if (this->type == "data") {
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if (this->type == "file") {
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if (this->type == "http") {
        av_dict_set(&opts, "timeout",          "10000000", 0);
        av_dict_set(&opts, "rw_timeout",       "10000000", 0);
        av_dict_set(&opts, "follow_redirects", "1",        0);
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if (this->type == "rtmp") {
        // 不能配置timeout
        av_dict_set(&opts, "rw_timeout", "10000000", 0);
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if (this->type == "rtsp") {
        av_dict_set(&opts, "timeout",    "10000000", 0);
        av_dict_set(&opts, "rw_timeout", "10000000", 0);
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if (this->type == "device") {
        #if CAIWEI_OS_WIN
        std::lock_guard<std::mutex> lock(device_mutex);
        if (device_open) {
            CW_LOG_W("设备已经打开: %s - %s", this->type.c_str(), this->url.c_str());
            ret = AVERROR(EBUSY);
        } else {
            open_device = true;
            device_open = true;
            av_dict_set(&opts, "framerate",         "30",       0);
            av_dict_set(&opts, "video_size",        "640x360",  0);
            av_dict_set(&opts, "pixel_format",      "yuyv422",  0);
            av_dict_set(&opts, "channels",          "1",        0);
            av_dict_set(&opts, "sample_rate",       "48000",    0);
            av_dict_set(&opts, "sample_size",       "16",       0);
            av_dict_set(&opts, "audio_buffer_size", "100",      0);
            const AVInputFormat* format = av_find_input_format("dshow");
            ret = avformat_open_input(&fmt_ctx, this->url.c_str(), format, &opts);
        }
        #else
        ret = AVERROR_PROTOCOL_NOT_FOUND;
        #endif
    } else {
        CW_LOG_W("不支持的协议: %s - %s", this->type.c_str(), this->url.c_str());
        ret = AVERROR_PROTOCOL_NOT_FOUND;
    }
    av_dict_free(&opts);
    if (ret != 0) {
        char err[AV_ERROR_MAX_STRING_SIZE]{};
        av_strerror(ret, err, sizeof(err));
        CW_LOG_W("媒体打开失败: %s - %s = %s", this->type.c_str(), this->url.c_str(), err);
        goto close_all;
    }
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        char err[AV_ERROR_MAX_STRING_SIZE]{};
        av_strerror(ret, err, sizeof(err));
        CW_LOG_W("媒体解析失败: %s - %s = %s", this->type.c_str(), this->url.c_str(), err);
        goto close_all;
    }
    for (uint32_t i = 0; i < fmt_ctx->nb_streams; ++i) {
        AVStream         * av_stream = fmt_ctx->streams[i];
        AVCodecParameters* codecpar  = av_stream->codecpar;
        if (codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            if (audio_stream_idx != -1) {
                continue;
            }
            if (!init_decoder_ctx(this->type, this->url, &audio_decoder_ctx, codecpar)) {
                CW_LOG_W("音频解码器上下文创建失败: %s - %s", this->type.c_str(), this->url.c_str());
                goto close_all;
            }
            audio_stream_idx = i;
            audio_time_base  = fmt_ctx->streams[i]->time_base;
        } else if (codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            if (video_stream_idx != -1) {
                continue;
            }
            if (!init_decoder_ctx(this->type, this->url, &video_decoder_ctx, codecpar)) {
                CW_LOG_W("视频解码器上下文创建失败: %s - %s", this->type.c_str(), this->url.c_str());
                goto close_all;
            }
            video_stream_idx = i;
            video_time_base  = fmt_ctx->streams[i]->time_base;
        } else {
            // -
        }
    }
    if (audio_stream_idx == -1 && video_stream_idx == -1) {
        CW_LOG_W("媒体接收失败: %s - %s", this->type.c_str(), this->url.c_str());
        goto close_all;
    }
    if (media_format != nullptr) {
        if (!media_format->open(audio_decoder_ctx, video_decoder_ctx, &audio_time_base, &video_time_base)) {
            CW_LOG_W("媒体格式化上下文创建失败: %s - %s", this->type.c_str(), this->url.c_str());
            goto close_all;
        }
    }
    audio_info.bytes_per_sample = av_get_bytes_per_sample((AVSampleFormat) audio_info.format);
    video_frame.channels = av_pix_fmt_desc_get((AVPixelFormat) video_info.format)->nb_components;
    while (this->running) {
        ret = av_read_frame(fmt_ctx, decoder_packet);
        if (ret == 0) {
            last_recv_time = std::chrono::steady_clock::now();
            if (decoder_packet->stream_index == audio_stream_idx) {
                if (media_format != nullptr) {
                    media_format->send(MediaType::AUDIO, decoder_packet);
                }
                msec = av_rescale_q(decoder_packet->pts, audio_time_base, msec_time_base);
                ret = avcodec_send_packet(audio_decoder_ctx, decoder_packet);
                while (ret == 0) {
                    ret = avcodec_receive_frame(audio_decoder_ctx, decoder_frame);
                    if (ret == 0) {
                        if (swr_ctx == nullptr) {
                            audio_frame_format      = decoder_frame->format;
                            audio_frame_ch_layout   = decoder_frame->ch_layout;
                            audio_frame_sample_rate = decoder_frame->sample_rate;
                            swr_ctx = init_audio_swr(audio_info, decoder_frame);
                            if (swr_ctx == nullptr) {
                                CW_LOG_W("音频重采样上下文创建失败: %s - %s", this->type.c_str(), this->url.c_str());
                                goto close_all;
                            }
                        } else if (
                            audio_frame_format                  != decoder_frame->format                ||
                            audio_frame_ch_layout.nb_channels   != decoder_frame->ch_layout.nb_channels ||
                            audio_frame_sample_rate             != decoder_frame->sample_rate
                        ) {
                            CW_LOG_I("音频格式发生变化: %s - %s", this->type.c_str(), this->url.c_str());
                            swr_free(&swr_ctx);
                            audio_frame_format      = decoder_frame->format;
                            audio_frame_ch_layout   = decoder_frame->ch_layout;
                            audio_frame_sample_rate = decoder_frame->sample_rate;
                            swr_ctx = init_audio_swr(audio_info, decoder_frame);
                            if (swr_ctx == nullptr) {
                                CW_LOG_W("音频重采样上下文创建失败: %s - %s", this->type.c_str(), this->url.c_str());
                                goto close_all;
                            }
                        }
                        if (recv_audio_frame(msec, audio_frames, swr_ctx, decoder_frame, audio_info, audio_frame)) {
                            if (audio_callback) {
                                if (audio_callback(audio_frame)) {
                                    last_send_time = std::chrono::steady_clock::now();
                                } else {
                                    CW_LOG_I("音频回调结束: %s - %s", this->type.c_str(), this->url.c_str());
                                    this->running = false;
                                }
                            }
                        }
                        ++audio_frames;
                    }
                    av_frame_unref(decoder_frame);
                }
            } else if (decoder_packet->stream_index == video_stream_idx) {
                if (media_format != nullptr) {
                    media_format->send(MediaType::VIDEO, decoder_packet);
                }
                msec = av_rescale_q(decoder_packet->pts, video_time_base, msec_time_base);
                ret = avcodec_send_packet(video_decoder_ctx, decoder_packet);
                while (ret == 0) {
                    ret = avcodec_receive_frame(video_decoder_ctx, decoder_frame);
                    if (ret == 0) {
                        if (sws_ctx == nullptr) {
                            video_frame_width  = decoder_frame->width;
                            video_frame_height = decoder_frame->height;
                            video_frame_format = decoder_frame->format;
                            sws_ctx = init_video_sws(video_info, decoder_frame);
                            if (sws_ctx == nullptr) {
                                CW_LOG_W("视频重采样上下文创建失败: %s - %s", this->type.c_str(), this->url.c_str());
                                goto close_all;
                            }
                        } else if (
                            video_frame_width  != decoder_frame->width  ||
                            video_frame_height != decoder_frame->height ||
                            video_frame_format != decoder_frame->format
                        ) {
                            CW_LOG_I("视频格式发生变化: %s - %s", this->type.c_str(), this->url.c_str());
                            sws_free(&sws_ctx);
                            video_frame_width  = decoder_frame->width;
                            video_frame_height = decoder_frame->height;
                            video_frame_format = decoder_frame->format;
                            sws_ctx = init_video_sws(video_info, decoder_frame);
                            if (sws_ctx == nullptr) {
                                CW_LOG_W("视频重采样上下文创建失败: %s - %s", this->type.c_str(), this->url.c_str());
                                goto close_all;
                            }
                        }
                        if (recv_video_frame(msec, video_frames, sws_ctx, decoder_frame, video_info, video_frame)) {
                            if (video_callback) {
                                if (video_callback(video_frame)) {
                                    last_send_time = std::chrono::steady_clock::now();
                                } else {
                                    CW_LOG_I("视频回调结束: %s - %s", this->type.c_str(), this->url.c_str());
                                    this->running = false;
                                }
                            }
                        }
                        ++video_frames;
                    }
                    av_frame_unref(decoder_frame);
                }
            } else {
                // -
            }
            av_packet_unref(decoder_packet);
            auto send_timeout = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_send_time).count();
            if (send_timeout >= timeout) {
                CW_LOG_W("媒体发送超时自动关闭: %s - %s = %d", this->type.c_str(), this->url.c_str(), send_timeout);
                goto close_all;
            }
        } else if (ret == AVERROR_EOF || ret == AVERROR_EXIT) {
            // 忽略刷出缓冲数据
            CW_LOG_I("媒体已经关闭: %s - %s = %d", this->type.c_str(), this->url.c_str(), ret);
            goto close_all;
        } else {
            CW_LOG_D("媒体接收失败: %s - %s = %d", this->type.c_str(), this->url.c_str(), ret);
            auto recv_timeout = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_recv_time).count();
            if (recv_timeout >= timeout) {
                CW_LOG_W("媒体接收超时自动关闭: %s - %s = %d", this->type.c_str(), this->url.c_str(), recv_timeout);
                goto close_all;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }
    close_all:
    this->running = false;
    av_frame_unref(decoder_frame);
    av_packet_unref(decoder_packet);
    av_frame_free(&decoder_frame);
    av_packet_free(&decoder_packet);
    if (swr_ctx) {
        swr_free(&swr_ctx);
    }
    if (sws_ctx) {
        sws_free(&sws_ctx);
    }
    if (fmt_ctx) {
        avformat_close_input(&fmt_ctx);
    }
    if (audio_decoder_ctx) {
        avcodec_free_context(&audio_decoder_ctx);
    }
    if (video_decoder_ctx) {
        avcodec_free_context(&video_decoder_ctx);
    }
    if (open_device && device_open) {
        std::lock_guard<std::mutex> lock(device_mutex);
        if (device_open) {
            device_open = false;
        }
    }
    if ((this->type == "rtp" || this->type == "sdp") && std::filesystem::exists(sdp_file)) {
        std::filesystem::remove(sdp_file);
    }
    CW_LOG_I("媒体接收线程结束: %s - %s = %d", this->type.c_str(), this->url.c_str(), ret);
    return true;
}

bool caiwei::media::MediaDemuxer::stop() {
    this->running = false;
    return true;
}

static bool write_sdp_file(const char* data, int size, const std::filesystem::path& path) {
    if (path.empty()) {
        CW_LOG_W("没有指定文件路径");
        return false;
    }
    std::filesystem::path parent = path.parent_path();
    if (!std::filesystem::exists(parent)) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream stream;
    stream.open(path, std::ios_base::binary);
    if (!stream.is_open()) {
        CW_LOG_W("保存文件失败: %s", path.string().c_str());
        stream.close();
        return false;
    }
    stream.write(data, size);
    stream.close();
    return true;
}

static SwrContext* init_audio_swr(caiwei::media::AudioInfo& audio_info, AVFrame* frame) {
    SwrContext* swr = swr_alloc();
    if (swr == nullptr) {
        CW_LOG_W("音频重采样打开失败");
        return nullptr;
    }
    AVChannelLayout ch_layout;
    av_channel_layout_default(&ch_layout, audio_info.channels);
    int ret = swr_alloc_set_opts2(
        &swr,
        &       ch_layout, (AVSampleFormat) audio_info.format, audio_info.sample_rate,
        &frame->ch_layout, (AVSampleFormat) frame->    format, frame->    sample_rate,
        0, nullptr
    );
    if (ret != 0) {
        swr_free(&swr);
        CW_LOG_W("音频重采样设置失败: %d", ret);
        return nullptr;
    }
    ret = swr_init(swr);
    if (ret != 0) {
        swr_free(&swr);
        CW_LOG_W("音频重采样加载失败: %d", ret);
        return nullptr;
    }
    return swr;
}

static SwsContext* init_video_sws(caiwei::media::VideoInfo& video_info, AVFrame* frame) {
    if (video_info.height <= 0) {
        video_info.height = video_info.width * frame->height / frame->width / 2 * 2;
    }
    SwsContext* sws = sws_getContext(
        frame->    width, frame->    height, (AVPixelFormat) frame->    format,
        video_info.width, video_info.height, (AVPixelFormat) video_info.format,
//      SWS_BILINEAR,
        SWS_FAST_BILINEAR,
        nullptr, nullptr, nullptr
    );
    if (sws == nullptr) {
        CW_LOG_W("视频重采样打开失败");
        return nullptr;
    }
    return sws;
}

static void sws_free(SwsContext** sws) {
    if (*sws) {
        sws_freeContext(*sws);
        *sws = nullptr;
    }
}

static bool init_decoder_ctx(const std::string& type, const std::string& url, AVCodecContext** decoder_ctx, AVCodecParameters* codecpar) {
    const AVCodec* decoder = avcodec_find_decoder(codecpar->codec_id);
    if (decoder == nullptr) {
        CW_LOG_W("解码器不存在: %s - %s", type.c_str(), url.c_str());
        return false;
    }
    *decoder_ctx = avcodec_alloc_context3(decoder);
    if (*decoder_ctx == nullptr) {
        CW_LOG_W("解码器创建失败: %s - %s", type.c_str(), url.c_str());
        return false;
    }
    int ret = avcodec_parameters_to_context(*decoder_ctx, codecpar);
    if (ret < 0) {
        CW_LOG_W("解码器设置失败: %s - %s = %d", type.c_str(), url.c_str(), ret);
        return false;
    }
    ret = avcodec_open2(*decoder_ctx, decoder, nullptr);
    if (ret != 0) {
        CW_LOG_W("解码器打开失败: %s - %s = %d", type.c_str(), url.c_str(), ret);
        return false;
    }
    if (codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
        CW_LOG_I("打开音频解码器: %s - %s = %s - %d - %d", type.c_str(), url.c_str(), decoder->name, codecpar->ch_layout.nb_channels, codecpar->sample_rate);
    } else if (codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        CW_LOG_I("打开视频解码器: %s - %s = %s - %d - %d", type.c_str(), url.c_str(), decoder->name, codecpar->width, codecpar->height);
    } else {
        // -
    }
    return true;
}
