#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/media.hpp"

#include <chrono>
#include <fstream>
#include <filesystem>

extern "C" {

#include "libavutil/opt.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
#include "libswresample/swresample.h"

}

static bool save_file(const char* data, int size, const std::filesystem::path path);

static SwrContext* init_audio_swr(caiwei::media::AudioInfo& audio_info, AVFrame* frame);
static SwsContext* init_video_sws(caiwei::media::VideoInfo& video_info, AVFrame* frame);

static void sws_free(SwsContext** sws);

caiwei::media::MediaDemuxer::MediaDemuxer(
    const std::string& type,
    const std::string& url,
    caiwei::media::AudioCallback audio_callback,
    caiwei::media::VideoCallback video_callback
) : running(false),
    type(type),
    url(url),
    audio_callback(audio_callback),
    video_callback(video_callback) {
}

caiwei::media::MediaDemuxer::~MediaDemuxer() {
    this->stop();
}

inline bool recv_audio_frame(size_t msec, size_t audio_frames, SwrContext* swr_ctx, AVFrame* frame, caiwei::media::AudioInfo& audio_info, caiwei::media::AudioFrame& audioFrame) {
    uint8_t* dst_data = audioFrame.data.data();
    int nb_samples = swr_get_out_samples(swr_ctx, frame->nb_samples);
        nb_samples = swr_convert(swr_ctx, &dst_data, nb_samples, (const uint8_t**) frame->data, frame->nb_samples);
    if (nb_samples < 0) {
        LOG_WARN("音频重采样失败: %d", nb_samples);
        return false;
    }
    audioFrame.msec        = msec;
    audioFrame.frames      = audio_frames;
    audioFrame.samples     = nb_samples;
    audioFrame.data_length = nb_samples * audio_info.channel * audio_info.bytes_per_sample;
    return true;
}

inline bool recv_video_frame(size_t msec, size_t video_frames, SwsContext* sws_ctx, AVFrame* frame, caiwei::media::VideoInfo& video_info, caiwei::media::VideoFrame& videoFrame) {
    uint8_t* dst_data   = videoFrame.data.data();
    int      dst_stride = video_info.width * 3;
    int height = sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, &dst_data, &dst_stride);
    if (height < 0) {
        LOG_WARN("视频重采样失败: %d", height);
        return false;
    }
    videoFrame.width       = video_info.width;
    videoFrame.height      = height;
    videoFrame.msec        = msec;
    videoFrame.frames      = video_frames;
    videoFrame.data_length = video_info.width * height * 3;
    return true;
}

bool caiwei::media::MediaDemuxer::open(AudioInfo audio_info, VideoInfo video_info) {
    // TODO 快速开关内存泄漏
    int ret = -1;
    std::string sdp_file; // RTP|SDP协议使用
    size_t audio_frames  = 0;
    size_t video_frames  = 0;
    int audio_stream_idx = -1;
    int video_stream_idx = -1;
    int             audio_frame_format      = 0;
    AVChannelLayout audio_frame_ch_layout;
    int             audio_frame_sample_rate = 0;
    int             video_frame_width       = 0;
    int             video_frame_height      = 0;
    int             video_frame_format      = 0;
    AVFrame * decoder_frame  = av_frame_alloc();
    AVPacket* decoder_packet = av_packet_alloc();
    AVDictionary   * opts    = nullptr;
    SwrContext     * swr_ctx = nullptr;
    SwsContext     * sws_ctx = nullptr;
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext * audio_decoder_ctx = nullptr;
    AVCodecContext * video_decoder_ctx = nullptr;
    const AVRational msec_base{ 1, 1000 };
    AVRational audio_time_base;
    AVRational video_time_base;
    auto timeout = caiwei::env::get_int("CAIWEI_TIMEOUT"); // 超时时间
    auto last_recv_time = std::chrono::system_clock::now(); // 最后接收时间
    auto last_send_time = std::chrono::system_clock::now(); // 最后发送时间
    caiwei::media::AudioFrame audioFrame(     256 * 1024);
    caiwei::media::VideoFrame videoFrame(8 * 1024 * 1024);
    av_dict_set(&opts, "seekable",          "0",                                          0);
    av_dict_set(&opts, "buffer_size",       "262144",                                     0);
    av_dict_set(&opts, "thread_queue_size", "2048",                                       0);
    av_dict_set(&opts, "protocol_whitelist", "tcp,tls,udp,rtp,file,http,rtmp,rtsp,https", 0);
    if(this->type == "rtp" || this->type == "sdp") {
        av_dict_set(&opts, "timeout",                     "10000000",               0);
        av_dict_set(&opts, "rw_timeout",                  "10000000",               0);
        av_dict_set(&opts, "fflags",                      "+genpts+discardcorrupt", 0);
        av_dict_set(&opts, "max_delay",                   "1000000",                0);
        av_dict_set(&opts, "reorder_queue_size",          "2048",                   0);
        av_dict_set(&opts, "use_wallclock_as_timestamps", "1",                      0);
        sdp_file = "./sdp/" + caiwei::env::id() + ".sdp";
        if (save_file(this->url.c_str(), this->url.size(), sdp_file)) {
            ret = avformat_open_input(&fmt_ctx, sdp_file.c_str(), nullptr, &opts);
        }
    } else if(this->type == "file") {
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if(this->type == "http") {
        av_dict_set(&opts, "timeout",          "10000000", 0);
        av_dict_set(&opts, "rw_timeout",       "10000000", 0);
        av_dict_set(&opts, "follow_redirects", "1",        0);
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if(this->type == "rtmp") {
        // 不能配置timeout
        av_dict_set(&opts, "rw_timeout", "10000000", 0);
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if(this->type == "rtsp") {
        av_dict_set(&opts, "timeout",    "10000000", 0);
        av_dict_set(&opts, "rw_timeout", "10000000", 0);
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if(this->type == "device") {
        // ffmpeg -list_devices true -f dshow -i dummy
        // url: audio=deviceName:video=deviceName
        #if OS_WIN
        av_dict_set(&opts, "framerate",         "30",       0);
        av_dict_set(&opts, "video_size",        "640*480",  0);
        av_dict_set(&opts, "pixel_format",      "yuyv422",  0);
        av_dict_set(&opts, "channels",          "1",        0);
        av_dict_set(&opts, "sample_rate",       "48000",    0);
        av_dict_set(&opts, "sample_size",       "16",       0);
        av_dict_set(&opts, "audio_buffer_size", "100",      0); // 音频缓冲ms
        const AVInputFormat* format = av_find_input_format("dshow");
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), format, &opts);
        #endif
    } else {
        LOG_WARN("不支持的协议：%s - %s", this->type.c_str(), this->url.c_str());
    }
    av_dict_free(&opts);
    if (ret != 0) {
        char err[AV_ERROR_MAX_STRING_SIZE]{};
        av_strerror(ret, err, sizeof(err));
        LOG_WARN("媒体打开失败：%s - %s = %s", this->type.c_str(), this->url.c_str(), err);
        goto close_all;
    }
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        char err[AV_ERROR_MAX_STRING_SIZE]{};
        av_strerror(ret, err, sizeof(err));
        LOG_WARN("媒体解析失败：%s - %s = %s", this->type.c_str(), this->url.c_str(), err);
        goto close_all;
    }
    for (uint32_t i = 0; i < fmt_ctx->nb_streams; ++i) {
        AVStream         * av_stream = fmt_ctx->streams[i];
        AVCodecParameters* codecpar  = av_stream->codecpar;
        if (codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx = i;
            const AVCodec* decoder = avcodec_find_decoder(codecpar->codec_id);
            audio_decoder_ctx = avcodec_alloc_context3(decoder);
            ret = avcodec_parameters_to_context(audio_decoder_ctx, codecpar);
            if(ret < 0) {
                LOG_WARN("音频解码器打开失败：%s - %s", this->type.c_str(), this->url.c_str());
                goto close_all;
            }
            ret = avcodec_open2(audio_decoder_ctx, decoder, nullptr);
            if(ret != 0) {
                LOG_WARN("音频解码器打开失败：%s - %s", this->type.c_str(), this->url.c_str());
                goto close_all;
            }
            audio_time_base = fmt_ctx->streams[i]->time_base;
            LOG_INFO("打开音频解码器：%s - %s = %s - %d - %d", this->type.c_str(), this->url.c_str(), decoder->name, codecpar->ch_layout.nb_channels, codecpar->sample_rate);
        } else if(codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            const AVCodec* decoder = avcodec_find_decoder(codecpar->codec_id);
            video_decoder_ctx = avcodec_alloc_context3(decoder);
            ret = avcodec_parameters_to_context(video_decoder_ctx, codecpar);
            if(ret < 0) {
                LOG_WARN("视频解码器打开失败：%s - %s", this->type.c_str(), this->url.c_str());
                goto close_all;
            }
            ret = avcodec_open2(video_decoder_ctx, decoder, nullptr);
            if(ret != 0) {
                LOG_WARN("视频解码器打开失败：%s - %s", this->type.c_str(), this->url.c_str());
                goto close_all;
            }
            video_time_base = fmt_ctx->streams[i]->time_base;
            LOG_INFO("打开视频解码器：%s - %s = %s - %d - %d", this->type.c_str(), this->url.c_str(), decoder->name, codecpar->width, codecpar->height);
        } else {
            // -
        }
    }
    if(audio_stream_idx == -1 && video_stream_idx == -1) {
        LOG_WARN("媒体接收失败：%s - %s", this->type.c_str(), this->url.c_str());
        goto close_all;
    }
    audio_info.bytes_per_sample = av_get_bytes_per_sample((AVSampleFormat) audio_info.format);
    this->running = true;
    while(this->running) {
        ret = av_read_frame(fmt_ctx, decoder_packet);
        if(ret == 0) {
            last_recv_time = std::chrono::system_clock::now();
            if (decoder_packet->stream_index == audio_stream_idx) {
                int64_t msec = av_rescale_q(decoder_packet->pts, audio_time_base, msec_base);
                // LOG_DEBUG("解码音频数据: %" PRId64 "ms - %" PRId64, msec, audio_frames);
                ret = avcodec_send_packet(audio_decoder_ctx, decoder_packet);
                while(ret == 0) {
                    ret = avcodec_receive_frame(audio_decoder_ctx, decoder_frame);
                    if(ret == 0) {
                        if(swr_ctx == nullptr) {
                            audio_frame_format      = decoder_frame->format;
                            audio_frame_ch_layout   = decoder_frame->ch_layout;
                            audio_frame_sample_rate = decoder_frame->sample_rate;
                            swr_ctx = init_audio_swr(audio_info, decoder_frame);
                        } else if (
                            audio_frame_format                  != decoder_frame->format                ||
                            audio_frame_ch_layout.nb_channels   != decoder_frame->ch_layout.nb_channels ||
                            audio_frame_sample_rate             != decoder_frame->sample_rate
                        ) {
                            LOG_WARN("音频格式变化: %s - %s", this->type.c_str(), this->url.c_str());
                            swr_free(&swr_ctx);
                            audio_frame_format      = decoder_frame->format;
                            audio_frame_ch_layout   = decoder_frame->ch_layout;
                            audio_frame_sample_rate = decoder_frame->sample_rate;
                            swr_ctx = init_audio_swr(audio_info, decoder_frame);
                        }
                        if (recv_audio_frame(msec, audio_frames, swr_ctx, decoder_frame, audio_info, audioFrame)) {
                            if (this->audio_callback) {
                                if (this->audio_callback(audioFrame)) {
                                    last_send_time = std::chrono::system_clock::now();
                                }
                            }
                        }
                        ++audio_frames;
                    }
                    av_frame_unref(decoder_frame);
                }
            } else if(decoder_packet->stream_index == video_stream_idx) {
                int64_t msec = av_rescale_q(decoder_packet->pts, video_time_base, msec_base);
                // LOG_DEBUG("解码视频数据: %" PRId64 "ms - %" PRId64, msec, video_frames);
                ret = avcodec_send_packet(video_decoder_ctx, decoder_packet);
                while(ret == 0) {
                    ret = avcodec_receive_frame(video_decoder_ctx, decoder_frame);
                    if(ret == 0) {
                        if(sws_ctx == nullptr) {
                            video_frame_width  = decoder_frame->width;
                            video_frame_height = decoder_frame->height;
                            video_frame_format = decoder_frame->format;
                            sws_ctx = init_video_sws(video_info, decoder_frame);
                        } else if(
                            video_frame_width  != decoder_frame->width  ||
                            video_frame_height != decoder_frame->height ||
                            video_frame_format != decoder_frame->format
                        ) {
                            LOG_WARN("视频格式变化: %s - %s", this->type.c_str(), this->url.c_str());
                            sws_free(&sws_ctx);
                            video_frame_width  = decoder_frame->width;
                            video_frame_height = decoder_frame->height;
                            video_frame_format = decoder_frame->format;
                            sws_ctx = init_video_sws(video_info, decoder_frame);
                        }
                        if (recv_video_frame(msec, video_frames, sws_ctx, decoder_frame, video_info, videoFrame)) {
                            if (this->video_callback) {
                                if (this->video_callback(videoFrame)) {
                                    last_send_time = std::chrono::system_clock::now();
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
            auto send_timeout = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - last_send_time).count();
            if(send_timeout >= timeout) {
                LOG_WARN("媒体发送超时自动关闭：%s - %s = %d", this->type.c_str(), this->url.c_str(), send_timeout);
                goto close_all;
            }
        } else if(ret == AVERROR_EOF || ret == AVERROR_EXIT) {
            LOG_INFO("媒体已经关闭: %s - %s = %d", this->type.c_str(), this->url.c_str(), ret);
            goto close_all;
        } else {
            LOG_INFO("读取媒体数据失败: %s - %s = %d", this->type.c_str(), this->url.c_str(), ret);
            auto recv_timeout = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - last_recv_time).count();
            if(recv_timeout >= timeout) {
                LOG_WARN("媒体接收超时自动关闭: %s - %s = %d", this->type.c_str(), this->url.c_str(), recv_timeout);
                goto close_all;
            }
        }
    }
    // TODO 释放顺序验证
    close_all:
    // 标记关闭
    this->running = false;
    // 释放资源
    av_frame_unref(decoder_frame);
    av_packet_unref(decoder_packet);
    av_frame_free(&decoder_frame);
    av_packet_free(&decoder_packet);
    if(swr_ctx) {
        swr_free(&swr_ctx);
    }
    if (sws_ctx) {
        sws_free(&sws_ctx);
    }
    if(fmt_ctx) {
        avformat_close_input(&fmt_ctx);
    }
    if(audio_decoder_ctx) {
        avcodec_free_context(&audio_decoder_ctx);
    }
    if(video_decoder_ctx) {
        avcodec_free_context(&video_decoder_ctx);
    }
    if(this->type == "rtp") {
        std::filesystem::remove(sdp_file);
    }
    LOG_INFO("媒体接收线程结束: %s - %s = %d", this->type.c_str(), this->url.c_str(), ret);
    return true;
}

bool caiwei::media::MediaDemuxer::stop() {
    this->running = false;
    return true;
}

static SwrContext* init_audio_swr(caiwei::media::AudioInfo& audio_info, AVFrame* frame) {
    SwrContext* swr = swr_alloc();
    if(swr == nullptr) {
        LOG_WARN("打开音频重采样失败");
        return nullptr;
    }
    AVChannelLayout ch_layout;
    av_channel_layout_default(&ch_layout, audio_info.channel);
    int ret = swr_alloc_set_opts2(
        &swr,
        &       ch_layout, (AVSampleFormat) audio_info.format, audio_info.sample_rate,
        &frame->ch_layout, (AVSampleFormat) frame->    format, frame->    sample_rate,
        0, nullptr
    );
    if(ret != 0) {
        swr_free(&swr);
        LOG_WARN("打开音频重采样失败: %d", ret);
        return nullptr;
    }
    ret = swr_init(swr);
    if(ret != 0) {
        swr_free(&swr);
        LOG_WARN("打开音频重采样失败: %d", ret);
        return nullptr;
    }
    return swr;
}

static SwsContext* init_video_sws(caiwei::media::VideoInfo& video_info, AVFrame* frame) {
    int height = video_info.height > 0 ? video_info.height : video_info.width * frame->height / frame->width / 2 * 2;
    SwsContext* sws = sws_getContext(
        frame->    width, frame->height, (AVPixelFormat) frame->    format,
        video_info.width,        height, (AVPixelFormat) video_info.format,
//      SWS_BILINEAR,
        SWS_FAST_BILINEAR,
        nullptr, nullptr, nullptr
    );
    if(sws == nullptr) {
        LOG_INFO("打开视频重采样失败");
        return nullptr;
    }
    return sws;
}

static void sws_free(SwsContext** sws) {
    if(*sws) {
        sws_freeContext(*sws);
        *sws = nullptr;
    }
}

static bool save_file(const char* data, int size, const std::filesystem::path path) {
    if(path.empty()) {
        LOG_WARN("没有指定文件路径");
        return false;
    }
    std::filesystem::path parent = path.parent_path();
    if(!std::filesystem::exists(parent)) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream stream;
    stream.open(path, std::ios_base::binary);
    if(!stream.is_open()) {
        LOG_WARN("保存文件失败: %s", path.string().c_str());
        stream.close();
        return false;
    }
    stream.write(data, size);
    stream.close();
    return true;
}
