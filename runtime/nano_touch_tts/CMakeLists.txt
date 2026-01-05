cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(nano_touch_tts VERSION 0.1)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(glog)
include(gflags)
include(gtest)
include(onnx)
include(llama.cpp)

add_subdirectory(src)
