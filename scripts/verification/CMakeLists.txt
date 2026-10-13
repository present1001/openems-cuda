cmake_minimum_required(VERSION 3.21)
project(openems_metal_backend_smoke LANGUAGES CXX OBJCXX)
if(NOT APPLE)
    message(FATAL_ERROR "Apple Metal smoke compilation requires macOS")
endif()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 11)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
find_path(BOOST_HEADERS boost/program_options.hpp
    HINTS /opt/homebrew/include /opt/homebrew/opt/boost/include /usr/local/include
    REQUIRED)
add_library(metal_backend_compile OBJECT
    ../../openEMS/FDTD/operator_metal.cpp
    ../../openEMS/FDTD/engine_metal.mm
)
target_include_directories(metal_backend_compile PRIVATE
    ../../openEMS
    ../../CSXCAD/src
    ../../fparser
    ${BOOST_HEADERS}
)
target_compile_options(metal_backend_compile PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
add_executable(metal_shader_smoke metal_shader_smoke.mm)
target_compile_options(metal_shader_smoke PRIVATE -fobjc-arc)
target_link_libraries(metal_shader_smoke PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)
enable_testing()
add_test(NAME metal_shader_runtime_compilation
    COMMAND metal_shader_smoke
    "${CMAKE_CURRENT_SOURCE_DIR}/../../openEMS/FDTD/engine_metal.mm"
)
