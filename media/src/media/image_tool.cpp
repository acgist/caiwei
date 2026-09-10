#include "caiwei/image_tool.hpp"

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE2_IMPLEMENTATION

#include "stb/stb_image.h"
#include "stb/stb_image_resize2.h"

#ifdef ENABLE_CAIWEI_TEST
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#endif
}
