# Disable glog features that don't work well on all platforms
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(WITH_GTEST OFF CACHE BOOL "" FORCE)

# Disable unwinder for Android/cross-compilation (no backtrace support)
if(CMAKE_CROSSCOMPILING OR ANDROID)
  set(WITH_UNWIND OFF CACHE BOOL "" FORCE)
  set(HAVE_LIB_UNWIND OFF CACHE BOOL "" FORCE)
endif()

# Use newer glog version with better Android support
FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/v0.6.0.zip
  URL_HASH SHA256=122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022
)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})
