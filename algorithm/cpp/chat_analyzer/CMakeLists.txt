cmake_minimum_required(VERSION 2.8)
project(chat_analyzer CXX)

# 기본값: Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# C++03 강제 (GCC/Clang 계열)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -std=gnu++98")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -g  -std=gnu++98")
endif()

# MSVC (VS2010) 최적화
if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(chan_demo
  src/main.cpp
)
