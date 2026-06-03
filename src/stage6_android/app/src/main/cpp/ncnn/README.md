# ncnn Android Prebuilt

This directory vendors the official ncnn Android prebuilt package:

- Source: https://github.com/Tencent/ncnn/releases/download/20260113/ncnn-20260113-android.zip
- Version: 20260113
- Included ABIs: arm64-v8a, armeabi-v7a, x86, x86_64

The Maven coordinate requested in the project notes (`com.tencent.ncnn:ncnn-android:1.0.20260526`) was not resolvable from the configured Android repositories, so CMake links this local official prebuilt package directly.
