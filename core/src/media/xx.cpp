#include <iostream>
#include <fstream>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libavutil/error.h>
}

#define SAMPLE_RATE     16000
#define CHANNELS        1
#define PCM_SAMPLE_FMT  AV_SAMPLE_FMT_S16
#define AAC_BIT_RATE    96000

static void print_err(int ret) {
    char buf[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(ret, buf, sizeof(buf));
    std::cerr << "ffmpeg err: " << buf << std::endl;
}

int main()
{
    const char* out_path = "output.m4a";
    const char* pcm_path = "input.pcm";

    // 1. 创建输出上下文
    AVFormatContext* fmt_ctx = nullptr;
    int ret = avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, out_path);
    if(ret < 0) { print_err(ret); return -1; }

    // 2. 新建音频流
    AVStream* audio_stream = avformat_new_stream(fmt_ctx, nullptr);
    if(!audio_stream) return -1;

    // 3. AAC编码器
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if(!codec) {
        std::cerr << "can not find aac encoder\n";
        return -1;
    }
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);

    codec_ctx->sample_rate = SAMPLE_RATE;
    codec_ctx->bit_rate = AAC_BIT_RATE;
    codec_ctx->channels = CHANNELS;
    codec_ctx->channel_layout = av_get_default_channel_layout(CHANNELS);
    // AAC编码器要求：FLTP 平面浮点
    codec_ctx->sample_fmt = AV_SAMPLE_FMT_FLTP;
    codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; // mp4封装必须

    ret = avcodec_open2(codec_ctx, codec, nullptr);
    if(ret < 0) { print_err(ret); return -1; }

    // 复制codec参数到stream
    ret = avcodec_parameters_from_context(audio_stream->codecpar, codec_ctx);
    if(ret <0) { print_err(ret); return -1; }

    // 4. swresample：s16le -> FLTP
    SwrContext* swr_ctx = swr_alloc();
    av_opt_set_int(swr_ctx, "in_sample_rate", SAMPLE_RATE, 0);
    av_opt_set_int(swr_ctx, "out_sample_rate", SAMPLE_RATE, 0);
    av_opt_set_int(swr_ctx, "in_channels", CHANNELS, 0);
    av_opt_set_int(swr_ctx, "out_channels", CHANNELS, 0);
    av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", PCM_SAMPLE_FMT, 0);
    av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_FLTP, 0);
    swr_init(swr_ctx);

    // 5. 打开输出文件
    ret = avio_open(&fmt_ctx->pb, out_path, AVIO_FLAG_WRITE);
    if(ret <0) { print_err(ret); return -1; }

    ret = avformat_write_header(fmt_ctx, nullptr);
    if(ret <0) { print_err(ret); return -1; }

    // AAC每一帧固定采样点数
    const int frame_size = codec_ctx->frame_size;
    AVFrame* frame = av_frame_alloc();
    frame->nb_samples = frame_size;
    frame->format = codec_ctx->sample_fmt;
    frame->sample_rate = SAMPLE_RATE;
    frame->channels = CHANNELS;
    frame->channel_layout = codec_ctx->channel_layout;
    av_frame_get_buffer(frame, 0);

    AVPacket* pkt = av_packet_alloc();

    // 读取PCM：s16le，每个采样2字节
    const int pcm_bytes_per_frame = frame_size * CHANNELS * 2;
    std::ifstream fin(pcm_path, std::ios::binary);
    if(!fin.is_open()) {
        std::cerr << "open input.pcm failed\n";
        return -1;
    }
    std::vector<uint8_t> pcm_buf(pcm_bytes_per_frame);

    // 循环编码
    while(true)
    {
        fin.read((char*)pcm_buf.data(), pcm_bytes_per_frame);
        auto read_len = fin.gcount();
        if(read_len <=0) break;

        // 如果最后一帧不足，补0
        if(read_len < pcm_bytes_per_frame) {
            memset(pcm_buf.data() + read_len, 0, pcm_bytes_per_frame - read_len);
        }

        // swr转换：s16le输入buffer，输出到frame
        const uint8_t* in_buf[1] = { pcm_buf.data() };
        swr_convert(swr_ctx,
            frame->data, frame_size,
            in_buf, frame_size
        );

        // 送入编码器
        ret = avcodec_send_frame(codec_ctx, frame);
        if(ret <0) { print_err(ret); continue; }

        while(true)
        {
            ret = avcodec_receive_packet(codec_ctx, pkt);
            if(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if(ret <0) { print_err(ret); goto end; }

            // 设置stream index
            pkt->stream_index = audio_stream->index;
            ret = av_interleaved_write_frame(fmt_ctx, pkt);
            av_packet_unref(pkt);
        }
    }

    // flush编码器剩余数据
    avcodec_send_frame(codec_ctx, nullptr);
    while(true)
    {
        ret = avcodec_receive_packet(codec_ctx, pkt);
        if(ret == AVERROR_EOF) break;
        if(ret <0) { print_err(ret); break; }
        pkt->stream_index = audio_stream->index;
        av_interleaved_write_frame(fmt_ctx, pkt);
        av_packet_unref(pkt);
    }

    av_write_trailer(fmt_ctx);

end:
    fin.close();
    av_packet_free(&pkt);
    av_frame_free(&frame);
    swr_free(&swr_ctx);
    avcodec_free_context(&codec_ctx);
    avio_closep(&fmt_ctx->pb);
    avformat_free_context(fmt_ctx);
    return 0;
}
