#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

#include "net.h"

namespace {

constexpr int kInputSize = 416;
constexpr const char* kInputBlob = "in0";
constexpr const char* kOutputBlob = "out0";

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "PigeonYolo", __VA_ARGS__)

struct Box {
  float left;
  float top;
  float right;
  float bottom;
  float score;
};

ncnn::Net g_net;
std::mutex g_net_mutex;
bool g_loaded = false;

float area(const Box& box) {
  return std::max(0.0f, box.right - box.left) * std::max(0.0f, box.bottom - box.top);
}

float intersection_over_union(const Box& a, const Box& b) {
  const float left = std::max(a.left, b.left);
  const float top = std::max(a.top, b.top);
  const float right = std::min(a.right, b.right);
  const float bottom = std::min(a.bottom, b.bottom);
  const float inter = area({left, top, right, bottom, 0.0f});
  const float union_area = area(a) + area(b) - inter;
  if (union_area <= 0.0f) return 0.0f;
  return inter / union_area;
}

std::vector<int> nms_sorted_boxes(const std::vector<Box>& boxes, float nms_threshold) {
  std::vector<int> picked;
  picked.reserve(boxes.size());
  for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
    bool keep = true;
    for (const int picked_index : picked) {
      if (intersection_over_union(boxes[i], boxes[picked_index]) > nms_threshold) {
        keep = false;
        break;
      }
    }
    if (keep) picked.push_back(i);
  }
  return picked;
}

int candidate_count(const ncnn::Mat& output) {
  const int total = static_cast<int>(output.total());
  if (total < 5) return 0;
  if (output.dims == 2) {
    if (output.h == 5) return output.w;
    if (output.w == 5) return output.h;
  }
  if (output.dims == 3) {
    if (output.c == 5) return output.w * output.h;
    if (output.h == 5) return output.w * output.c;
    if (output.w == 5) return output.h * output.c;
  }
  return total / 5;
}

float output_value(const ncnn::Mat& output, int index, int field, int count) {
  const float* data = output;
  if (output.dims == 2) {
    if (output.h == 5) return output.row(field)[index];
    if (output.w == 5) return output.row(index)[field];
  }
  if (output.dims == 3) {
    if (output.c == 5) {
      const float* channel = output.channel(field);
      return channel[index];
    }
    if (output.h == 5) {
      const int channel_index = index / output.w;
      const int x = index % output.w;
      const float* channel = output.channel(channel_index);
      return channel[field * output.w + x];
    }
    if (output.w == 5) {
      return data[index * 5 + field];
    }
  }
  return data[field * count + index];
}

std::vector<Box> parse_output(
    const ncnn::Mat& output,
    float confidence_threshold,
    int image_width,
    int image_height) {
  const int count = candidate_count(output);
  std::vector<Box> boxes;
  boxes.reserve(count / 4);

  for (int i = 0; i < count; ++i) {
    const float confidence = output_value(output, i, 4, count);
    if (confidence < confidence_threshold) continue;

    float cx = output_value(output, i, 0, count);
    float cy = output_value(output, i, 1, count);
    float width = output_value(output, i, 2, count);
    float height = output_value(output, i, 3, count);
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(width) || !std::isfinite(height)) continue;
    if (width <= 0.0f || height <= 0.0f) continue;

    const bool normalized = cx <= 1.5f && cy <= 1.5f && width <= 1.5f && height <= 1.5f;
    float left;
    float top;
    float right;
    float bottom;
    if (normalized) {
      left = (cx - width * 0.5f) * image_width;
      top = (cy - height * 0.5f) * image_height;
      right = (cx + width * 0.5f) * image_width;
      bottom = (cy + height * 0.5f) * image_height;
    } else {
      const float scale_x = static_cast<float>(image_width) / kInputSize;
      const float scale_y = static_cast<float>(image_height) / kInputSize;
      left = (cx - width * 0.5f) * scale_x;
      top = (cy - height * 0.5f) * scale_y;
      right = (cx + width * 0.5f) * scale_x;
      bottom = (cy + height * 0.5f) * scale_y;
    }

    left = std::clamp(left, 0.0f, static_cast<float>(image_width - 1));
    top = std::clamp(top, 0.0f, static_cast<float>(image_height - 1));
    right = std::clamp(right, left + 1.0f, static_cast<float>(image_width));
    bottom = std::clamp(bottom, top + 1.0f, static_cast<float>(image_height));
    boxes.push_back({left, top, right, bottom, confidence});
  }

  std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
  return boxes;
}

bool copy_rgba_pixels(JNIEnv* env, jobject bitmap, AndroidBitmapInfo* info, std::vector<unsigned char>* rgba) {
  void* pixels = nullptr;
  if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
    LOGE("AndroidBitmap_lockPixels failed");
    return false;
  }

  const int row_bytes = static_cast<int>(info->width) * 4;
  const auto* src = static_cast<const unsigned char*>(pixels);
  rgba->resize(static_cast<size_t>(row_bytes) * info->height);
  if (static_cast<int>(info->stride) == row_bytes) {
    std::memcpy(rgba->data(), src, rgba->size());
  } else {
    for (uint32_t y = 0; y < info->height; ++y) {
      std::memcpy(rgba->data() + static_cast<size_t>(y) * row_bytes, src + static_cast<size_t>(y) * info->stride, row_bytes);
    }
  }

  AndroidBitmap_unlockPixels(env, bitmap);
  return true;
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_pigeonvision_app_YoloDetector_nativeLoadModel(JNIEnv* env, jobject /* thiz */, jstring param_path, jstring bin_path) {
  const char* param = env->GetStringUTFChars(param_path, nullptr);
  const char* bin = env->GetStringUTFChars(bin_path, nullptr);
  if (param == nullptr || bin == nullptr) {
    if (param != nullptr) env->ReleaseStringUTFChars(param_path, param);
    if (bin != nullptr) env->ReleaseStringUTFChars(bin_path, bin);
    return JNI_FALSE;
  }

  std::lock_guard<std::mutex> lock(g_net_mutex);
  g_net.clear();
  g_net.opt.num_threads = 2;
  g_net.opt.use_vulkan_compute = false;
  const int param_result = g_net.load_param(param);
  const int model_result = param_result == 0 ? g_net.load_model(bin) : -1;
  g_loaded = param_result == 0 && model_result == 0;

  env->ReleaseStringUTFChars(param_path, param);
  env->ReleaseStringUTFChars(bin_path, bin);
  if (!g_loaded) {
    LOGE("Failed to load YOLO model: param=%d model=%d", param_result, model_result);
  }
  return g_loaded ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_pigeonvision_app_YoloDetector_nativeDetect(
    JNIEnv* env,
    jobject /* thiz */,
    jobject bitmap,
    jfloat confidence_threshold,
    jfloat nms_threshold) {
  AndroidBitmapInfo info;
  if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
    LOGE("AndroidBitmap_getInfo failed");
    return nullptr;
  }
  if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || info.width == 0 || info.height == 0) {
    LOGE("Unsupported bitmap format: %d", info.format);
    return nullptr;
  }

  std::vector<unsigned char> rgba;
  if (!copy_rgba_pixels(env, bitmap, &info, &rgba)) {
    return nullptr;
  }

  ncnn::Mat input =
      ncnn::Mat::from_pixels_resize(rgba.data(), ncnn::Mat::PIXEL_RGBA2RGB, info.width, info.height, kInputSize, kInputSize);
  const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
  const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  input.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Mat output;
  {
    std::lock_guard<std::mutex> lock(g_net_mutex);
    if (!g_loaded) return nullptr;
    ncnn::Extractor extractor = g_net.create_extractor();
    if (extractor.input(kInputBlob, input) != 0) {
      LOGE("Failed to bind input blob");
      return nullptr;
    }
    if (extractor.extract(kOutputBlob, output) != 0) {
      LOGE("Failed to extract output blob");
      return nullptr;
    }
  }

  std::vector<Box> boxes = parse_output(output, confidence_threshold, info.width, info.height);
  if (boxes.empty()) return nullptr;
  const std::vector<int> picked = nms_sorted_boxes(boxes, nms_threshold);
  if (picked.empty()) return nullptr;

  const Box& best = boxes[picked.front()];
  const jfloat values[5] = {best.left, best.top, best.right, best.bottom, best.score};
  jfloatArray result = env->NewFloatArray(5);
  if (result == nullptr) return nullptr;
  env->SetFloatArrayRegion(result, 0, 5, values);
  return result;
}
