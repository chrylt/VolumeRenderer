cmake_minimum_required(VERSION 3.10)
project(VolumeRenderer)

# Add OpenVDB Library
find_package(OpenVDB REQUIRED)

# Add Basalt submodule
add_subdirectory(external/Basalt)

# Add Shader directory
add_subdirectory(shaders)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/Basalt/include
)

# Source files
set(SOURCES
    src/main.cpp
    # Add other source files here
)

# Create executable
add_executable(VolumeRenderer ${SOURCES})

# Link Basalt library
target_link_libraries(VolumeRenderer PRIVATE Basalt)

# Set C++ standard
set_target_properties(VolumeRenderer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
