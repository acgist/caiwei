#include "test_runtime.hpp"

#include "caiwei/media.hpp"
#include "caiwei/context.hpp"

#include "caiwei/transform.hpp"

[[maybe_unused]]
void det_image() {
    // CUDA    100 =  1777 ms
    // Debug   100 = 19169 ms
    // Release 100 = 10160 ms
    int width, height, channels;
    auto data = stbi_load("./caiwei.jpg", &width, &height, &channels, STBI_default);
    caiwei::media::ImageFrame frame(width * height * channels);
    std::copy_n(data, width * height * channels, frame.data.data());
    frame.width = width;
    frame.height = height;
    frame.channels = channels;
    auto ptr = caiwei::context::get_context<caiwei::context::DetContext>(caiwei::context::Type::YOLO);
    CAIWEI_FOR_EACH(100)
    auto result = ptr->ptr()->run(frame);
    // caiwei::media::draw();
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

int main() {
    init_test();
    det_image();
    stop_test();
    return 0;
}
