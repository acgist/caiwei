#include "caiwei/env.hpp"
#include "caiwei/media.hpp"

#include <chrono>

extern "C" {

#include "libavutil/opt.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
#include "libswresample/swresample.h"

}

const static AVSampleFormat  audio_format           = AV_SAMPLE_FMT_S16;
const static AVChannelLayout audio_layout           = AV_CHANNEL_LAYOUT_MONO;
const static int             audio_nb_channels      = 1;
const static int             audio_sample_rate      = 16000;
const static int             audio_bytes_per_sample = av_get_bytes_per_sample(audio_format);

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
}


inline void recv_audio_frame(size_t msec, size_t audio_frames, SwrContext* swr_ctx, AVFrame* frame, caiwei::media::AudioFrame& aFrame) {
    int out_samples = swr_get_out_samples(swr_ctx, frame->nb_samples);
    auto* buffer_ptr = aFrame.data.data();
    out_samples = swr_convert(swr_ctx, &buffer_ptr, out_samples, (const uint8_t**) frame->data, frame->nb_samples);
    if(out_samples < 0) {
        // SPDLOG_ERROR("音频重采样失败：{}", out_samples);
        return;
    }
    aFrame.data_length = audio_nb_channels * audio_bytes_per_sample * out_samples;
    aFrame.msec = msec;
    aFrame.frames = audio_frames;
}

inline void recv_video_frame(size_t msec, size_t video_frames, SwsContext* sws_ctx, AVFrame* frame, caiwei::media::VideoFrame& vFrame) {
    int w = frame->width;
    int h = frame->height;
    uint8_t* dst_data   = vFrame.data.data();
    int      dst_stride = h;
    int out_heigth = sws_scale(sws_ctx, frame->data, frame->linesize, 0, h, &dst_data, &dst_stride);
    vFrame.data_length = w * h * 3;
    vFrame.width = w;
    vFrame.height = h;
    vFrame.msec = msec;
    vFrame.frames = video_frames;
}

bool caiwei::media::MediaDemuxer::open() {
    int ret = -1;
    int video_fps    = 0;
    int video_width  = 0;
    int video_height = 0;
    int frame_width  = 0;
    int frame_height = 0;
    size_t audio_frames = 0;
    size_t video_frames = 0;
    int audio_stream_idx = -1;
    int video_stream_idx = -1;
    AVFrame * decoder_frame  = av_frame_alloc();
    AVPacket* decoder_packet = av_packet_alloc();
    AVDictionary   * opts    = nullptr;
    SwrContext     * swr_ctx = nullptr;
    SwsContext     * sws_ctx = nullptr;
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext * audio_decoder_ctx = nullptr;
    AVCodecContext * video_decoder_ctx = nullptr;
    auto last_recv_time = std::chrono::system_clock::now(); // 最后接收时间
    auto last_send_time = std::chrono::system_clock::now(); // 最后发送时间
    auto timeout = std::atoi(caiwei::env::get("CAIWEI_TIMEOUT").c_str()); // 超时时间
    caiwei::media::AudioFrame aFrame(     256 * 1024);
    caiwei::media::VideoFrame vFrame(4 * 1024 * 1024);
    av_dict_set(&opts, "seekable",          "0",                                0);
    av_dict_set(&opts, "buffer_size",       "262144",                           0);
    av_dict_set(&opts, "thread_queue_size", "2048",                             0);
    av_dict_set(&opts, "protocol_whitelist", "tcp,udp,rtp,file,http,rtmp,rtsp", 0);
    if(this->type == "file") {
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if(this->type == "http") {
        av_dict_set(&opts, "timeout",          "10000000", 0);
        av_dict_set(&opts, "rw_timeout",       "30000000", 0);
        av_dict_set(&opts, "follow_redirects", "1",        0);
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if(this->type == "rtp") {
        av_dict_set(&opts, "timeout",                     "10000000",               0);
        av_dict_set(&opts, "rw_timeout",                  "30000000",               0);
        av_dict_set(&opts, "fflags",                      "+genpts+discardcorrupt", 0);
        av_dict_set(&opts, "rtbufsize",                   "8M",                     0);
        av_dict_set(&opts, "max_delay",                   "1000000",                0);
        av_dict_set(&opts, "reorder_queue_size",          "2048",                   0);
        av_dict_set(&opts, "use_wallclock_as_timestamps", "1",                      0);
        // const std::string sdp_file = SDP_FILE_PATH(stream->id);
        // aliang::saveFile(stream->sdp.c_str(), stream->sdp.size(), { sdp_file });
        // ret = avformat_open_input(&fmt_ctx, sdp_file.c_str(), nullptr, &opts);
    } else if(this->type == "rtsp") {
        av_dict_set(&opts, "timeout",    "10000000", 0);
        av_dict_set(&opts, "rw_timeout", "30000000", 0);
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if(this->type == "rtmp") {
        // 不能配置timeout
        av_dict_set(&opts, "rw_timeout", "30000000", 0);
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), nullptr, &opts);
    } else if(this->type == "device") {
        // ffmpeg -list_devices true -f dshow -i dummy
        #if OS_WIN
        av_dict_set(&opts, "framerate",    "30",       0);
        av_dict_set(&opts, "rtbufsize",    "8M",       0);
        av_dict_set(&opts, "video_size",   "640*480",  0);
        av_dict_set(&opts, "pixel_format", "yuyv422",  0);
        const AVInputFormat* fmt_dshow = av_find_input_format("dshow");
        ret = avformat_open_input(&fmt_ctx, this->url.c_str(), fmt_dshow, &opts);
        #endif
    } else {
        av_dict_free(&opts);
        // SPDLOG_WARN("不支持的协议：{} - {}", this->url, this->type);
        goto close_all;
    }
    av_dict_free(&opts);
    if (ret != 0) {
        // SPDLOG_WARN("媒体打开失败：{} - {} - {}", stream->id, stream->type, stream->url);
        goto close_all;
    }
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        // SPDLOG_WARN("媒体解析失败：{} - {} - {}", stream->id, stream->type, stream->url);
        goto close_all;
    }
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        AVStream         * av_stream = fmt_ctx->streams[i];
        AVCodecParameters* codecpar  = av_stream->codecpar;
        if (codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx = i;
            // 打开音频解码器
            const AVCodec* decoder = avcodec_find_decoder(codecpar->codec_id);
            audio_decoder_ctx = avcodec_alloc_context3(decoder);
            ret = avcodec_parameters_to_context(audio_decoder_ctx, codecpar);
            if(ret < 0) {
                // SPDLOG_WARN("音频参数转换失败：{} - {} - {}", stream->id, stream->type, stream->url);
                goto close_all;
            }
            ret = avcodec_open2(audio_decoder_ctx, decoder, nullptr);
            if(ret != 0) {
                // SPDLOG_WARN("音频解码器打开失败：{} - {} - {}", stream->id, stream->type, stream->url);
                goto close_all;
            }
            // SPDLOG_INFO("音频解码器：{} - {}", stream->id, decoder->name);
        } else if(codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            video_fps    = av_q2d(av_stream->r_frame_rate);
            video_width  = codecpar->width;
            video_height = codecpar->height;
            // 打开视频解码器
            const AVCodec* decoder = avcodec_find_decoder(codecpar->codec_id);
            video_decoder_ctx = avcodec_alloc_context3(decoder);
            ret = avcodec_parameters_to_context(video_decoder_ctx, codecpar);
            if(ret < 0) {
                // SPDLOG_WARN("视频参数转换失败：{} - {} - {}", stream->id, stream->type, stream->url);
                goto close_all;
            }
            ret = avcodec_open2(video_decoder_ctx, decoder, nullptr);
            if(ret != 0) {
                // SPDLOG_WARN("视频解码器打开失败：{} - {} - {}", stream->id, stream->type, stream->url);
                goto close_all;
            }
            // SPDLOG_INFO("视频解码器：{} - {} - {} - {} - {}", stream->id, decoder->name, video_fps, video_width, video_height);
        } else {
            // -
        }
    }
    if(audio_stream_idx == -1 && video_stream_idx == -1) {
        // SPDLOG_WARN("媒体接收失败：{} - {} - {}", stream->id, stream->type, stream->url);
        goto close_all;
    }
    // video_fps    = get_default_value(video_fps,    0,   32, ALIANG_DEFAULT_FPS   );
    // video_width  = get_default_value(video_width,  0, 1920, ALIANG_DEFAULT_WIDTH );
    // video_height = get_default_value(video_height, 0, 1920, ALIANG_DEFAULT_HEIGHT);
    // SPDLOG_INFO("视频帧率：{} - {}", stream->id, video_fps   );
    // SPDLOG_INFO("视频宽度：{} - {}", stream->id, video_width );
    // SPDLOG_INFO("视频高度：{} - {}", stream->id, video_height);
    this->running = true;
    while(this->running) {
        const AVRational msecbase{ 1, 1000 };
        ret = av_read_frame(fmt_ctx, decoder_packet);
        if(ret == 0) {
            last_recv_time = std::chrono::system_clock::now();
            const AVRational& timebase = fmt_ctx->streams[decoder_packet->stream_index]->time_base;
            int64_t msec = av_rescale_q(decoder_packet->pts, timebase, msecbase);
//          SPDLOG_DEBUG("解码数据：{} - {} - {} - {}", stream->id, msec, audio_frames, video_frames);
            if (decoder_packet->stream_index == audio_stream_idx) {
                ret = avcodec_send_packet(audio_decoder_ctx, decoder_packet);
                while(ret == 0) {
                    ret = avcodec_receive_frame(audio_decoder_ctx, decoder_frame);
                    if(ret == 0) {
                        if(swr_ctx == nullptr) {
                            swr_ctx = init_audio_swr(audio_decoder_ctx, decoder_frame);
                        }
                        // TODO resize
                        recv_audio_frame(msec, audio_frames, swr_ctx, decoder_frame, aFrame);
                        if (this->audio_callback) {
                            this->audio_callback(aFrame);
                        }
                        ++audio_frames;
                    }
                    av_frame_unref(decoder_frame);
                }
            } else if(decoder_packet->stream_index == video_stream_idx) {
                ret = avcodec_send_packet(video_decoder_ctx, decoder_packet);
                while(ret == 0) {
                    ret = avcodec_receive_frame(video_decoder_ctx, decoder_frame);
                    if(ret == 0) {
                        if(sws_ctx == nullptr) {
                            frame_width  = decoder_frame->width;
                            frame_height = decoder_frame->height;
                            sws_ctx = init_video_sws(video_decoder_ctx, decoder_frame);
                        } else if(frame_width != decoder_frame->width || frame_height != decoder_frame->height) {
                            // SPDLOG_INFO("视频大小变化：{} - {} - {} - {} - {}", stream->id, frame_width, frame_height, decoder_frame->width, decoder_frame->height);
                            sws_free(&sws_ctx);
                            frame_width  = decoder_frame->width;
                            frame_height = decoder_frame->height;
                            sws_ctx = init_video_sws(video_decoder_ctx, decoder_frame);
                        }
                        recv_video_frame(msec, video_frames, sws_ctx, decoder_frame, vFrame);
                        if (this->video_callback) {
                            this->video_callback(vFrame);
                        }
                        ++video_frames;
                    }
                    av_frame_unref(decoder_frame);
                }
            } else {
                // -
            }
            av_packet_unref(decoder_packet);
            // // 判断发送超时
            // if(stream->subscriber.empty()) {
            //     auto send_timeout = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - last_send_time).count();
            //     if(send_timeout >= timeout) {
            //         // 长时间没有消费数据关闭媒体
            //         // SPDLOG_INFO("媒体发送超时自动关闭：{} - {}", stream->id, send_timeout);
            //         goto close_all;
            //     }
            // } else {
            //     last_send_time = std::chrono::system_clock::now();
            // }
        } else if(ret == AVERROR_EOF || ret == AVERROR_EXIT) {
            // SPDLOG_INFO("媒体已经关闭：{}", stream->id);
            goto close_all;
        } else {
            // SPDLOG_INFO("读取媒体数据失败：{} - {}", stream->id, ret);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto recv_timeout = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - last_recv_time).count();
            if(recv_timeout >= timeout) {
                // 长时间没有接收数据关闭媒体
                // SPDLOG_INFO("媒体接收超时自动关闭：{} - {}", stream->id, recv_timeout);
                goto close_all;
            }
        }
    }
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
    // 注意：必须释放资源后才删除
    // if(this->type == "rtp") {
    //     aliang::deleteFile({ SDP_FILE_PATH(stream->id) });
    // }
    // SPDLOG_INFO("接收线程结束：{} - {} - {}", stream->id, stream->recv_queue.size(), stream->send_queue.size());
    return true;
}

bool caiwei::media::MediaDemuxer::stop() {
    this->running = false;
    if (thread.joinable()) {
        thread.join();
    }
    return true;
}
