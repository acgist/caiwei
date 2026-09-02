/**
 * 播放器
 */
#ifndef CAIWEI_PLAYER_HPP
#define CAIWEI_PLAYER_HPP

namespace caiwei {
namespace player {

bool open_player(int sample_rate, int nb_channels, int video_width, int video_height);
void stop_player();

bool play_audio(const void* data, int len);
bool play_video(const void* data, int len);

} // player
} // caiwei

#endif // CAIWEI_PLAYER_HPP
