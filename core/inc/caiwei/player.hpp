/**
 * 播放器
 */
#ifndef CAIWEI_PLAYER_HPP
#define CAIWEI_PLAYER_HPP

namespace caiwei {
namespace player {

extern bool open_player(int sample_rate, int nb_channels, int video_width, int video_height);
extern void stop_player();

extern bool play_audio(const void* data, int len);
extern bool play_video(const void* data, int len);

} // player
} // chobits

#endif // CAIWEI_PLAYER_HPP
