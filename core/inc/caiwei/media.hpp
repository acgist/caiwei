/**
 * 媒体工具
 * 
 * 提供音频、图片、视频处理功能
 * 
 * 媒体：采集
 * 音频：解码、重采样
 * 视频：解码、重采样
 * 图片：解码、缩放、画框、范围框选拷贝
 * 
 * OpenCV性能比较
 */
#ifndef CAIWEI_MEDIA_HPP
#define CAIWEI_MEDIA_HPP

namespace caiwei {
namespace media  {

/**
 * 媒体帧
 */
class Frame {};

// AudioFrame 数据 时间 时长
class AudioFrame : public Frame {};

// ImageFrame 数据 缩放 画框 框选拷贝
class ImageFrame : public Frame {};

// VideoFrame extend ImageFrame 时间 时长
class VideoFrame : public Frame {};

/**
 * 媒体封装器
 */
class MediaMuxer {};

/**
 * 媒体解封装器
 */
class MediaDemuxer {};

} // namespace media
} // namespace caiwei
#endif // CAIWEI_MEDIA_HPP
