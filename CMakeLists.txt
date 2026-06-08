cmake_minimum_required(VERSION 3.20)
project(vecsearch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -g in release so a profiler shows function names. Costs nothing at runtime.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")

option(SANITIZE "build with address and undefined behaviour sanitizers" OFF)
if(SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(vecsearch src/hnsw.cpp src/dataset.cpp src/eval.cpp)
target_include_directories(vecsearch PUBLIC src)
target_compile_options(vecsearch PRIVATE -Wall -Wextra)

add_executable(test tests/test.cpp)
target_link_libraries(test PRIVATE vecsearch)

add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE vecsearch)

add_executable(draw tools/draw.cpp)
target_link_libraries(draw PRIVATE vecsearch)

add_executable(trace tools/trace.cpp)
target_link_libraries(trace PRIVATE vecsearch)
