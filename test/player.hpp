#ifndef CAIWEI_TEST_PLAYER_HPP
#define CAIWEI_TEST_PLAYER_HPP

namespace caiwei {
namespace player {

bool open_player(int nb_channels = 1, int sample_rate = 16000, int video_width = 640, int video_height = 360);
void stop_player();

bool play_audio(const void* data, int len);
bool play_video(const void* data, int len);

} // player
} // caiwei

#endif // CAIWEI_TEST_PLAYER_HPP
