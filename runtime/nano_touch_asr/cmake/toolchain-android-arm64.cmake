# Cross-compilation toolchain for Android ARM64 (aarch64)
# Usage: cmake -B build-android -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-android-arm64.cmake
#
# Prerequisites:
#   - Android NDK installed
#   - Set ANDROID_NDK environment variable or pass -DANDROID_NDK=/path/to/ndk
#
# Example:
#   export ANDROID_NDK=$HOME/Library/Android/sdk/ndk/26.1.10909125
#   cmake -B build-android -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-android-arm64.cmake
#   cmake --build build-android

# Find Android NDK
if(NOT DEFINED ANDROID_NDK)
  if(DEFINED ENV{ANDROID_NDK})
    set(ANDROID_NDK $ENV{ANDROID_NDK})
  elseif(DEFINED ENV{ANDROID_NDK_HOME})
    set(ANDROID_NDK $ENV{ANDROID_NDK_HOME})
  elseif(EXISTS "$ENV{HOME}/Library/Android/sdk/ndk")
    # macOS default Android Studio NDK location
    file(GLOB NDK_VERSIONS "$ENV{HOME}/Library/Android/sdk/ndk/*")
    list(SORT NDK_VERSIONS ORDER DESCENDING)
    list(GET NDK_VERSIONS 0 ANDROID_NDK)
  elseif(EXISTS "$ENV{HOME}/Android/Sdk/ndk")
    # Linux default Android Studio NDK location
    file(GLOB NDK_VERSIONS "$ENV{HOME}/Android/Sdk/ndk/*")
    list(SORT NDK_VERSIONS ORDER DESCENDING)
    list(GET NDK_VERSIONS 0 ANDROID_NDK)
  endif()
endif()

if(NOT ANDROID_NDK)
  message(FATAL_ERROR "ANDROID_NDK not found. Please set ANDROID_NDK environment variable or pass -DANDROID_NDK=/path/to/ndk")
endif()

message(STATUS "Using Android NDK: ${ANDROID_NDK}")

# Android settings
set(ANDROID_ABI arm64-v8a)
set(ANDROID_PLATFORM android-24)  # Android 7.0+
set(ANDROID_STL c++_shared)

# Use Android NDK's toolchain file
include(${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
