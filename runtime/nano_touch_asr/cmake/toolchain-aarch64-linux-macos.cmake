# Cross-compilation toolchain for ARM64 (aarch64) Linux on macOS
# Usage: cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-linux-macos.cmake
#
# Prerequisites:
#   brew tap messense/macos-cross-toolchains
#   brew install aarch64-unknown-linux-gnu

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Homebrew prefix (handles both Intel and Apple Silicon Macs)
if(EXISTS "/opt/homebrew")
  set(HOMEBREW_PREFIX "/opt/homebrew")
else()
  set(HOMEBREW_PREFIX "/usr/local")
endif()

# Toolchain paths from Homebrew
set(TOOLCHAIN_PREFIX "aarch64-unknown-linux-gnu")
set(TOOLCHAIN_PATH "${HOMEBREW_PREFIX}/bin")

# Specify the cross compiler
set(CMAKE_C_COMPILER "${TOOLCHAIN_PATH}/${TOOLCHAIN_PREFIX}-gcc")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_PATH}/${TOOLCHAIN_PREFIX}-g++")
set(CMAKE_AR "${TOOLCHAIN_PATH}/${TOOLCHAIN_PREFIX}-ar")
set(CMAKE_RANLIB "${TOOLCHAIN_PATH}/${TOOLCHAIN_PREFIX}-ranlib")
set(CMAKE_STRIP "${TOOLCHAIN_PATH}/${TOOLCHAIN_PREFIX}-strip")

# Where to look for the target environment
set(CMAKE_FIND_ROOT_PATH "${HOMEBREW_PREFIX}/${TOOLCHAIN_PREFIX}")

# Search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Compiler flags for ARM64 optimization
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
