cmake_minimum_required(VERSION 3.22.1)

project(pigeon_yolo)

add_library(pigeon_yolo SHARED yolo_jni.cpp)

set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn")
find_package(ncnn REQUIRED CONFIG)

target_link_libraries(
    pigeon_yolo
    ncnn
    android
    log
)
