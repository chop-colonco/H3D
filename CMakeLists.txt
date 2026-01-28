cmake_minimum_required(VERSION 3.10)
project(metropolis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use the libtorch folder inside this project (NO "~" in CMake)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(metropolis
  main_parallel.cpp
  Metropolis.cpp
  helpers.cpp
)

target_link_libraries(metropolis
  PRIVATE
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Recommended by libtorch to get proper flags on some setups
target_compile_options(metropolis PRIVATE ${TORCH_CXX_FLAGS})
