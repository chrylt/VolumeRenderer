cmake_minimum_required(VERSION 3.10)
project(VolumeRenderer)

# Add Basalt submodule
add_subdirectory(external/Basalt)

# Add OpenVDB Library
find_package(OpenVDB REQUIRED)

# Find other packages (excluding GLM)
find_package(glfw3 REQUIRED)
find_package(stb REQUIRED)
find_package(assimp REQUIRED)
find_package(Vulkan REQUIRED)

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
add_executable(VolumeRenderer ${SOURCES} "include/ray_tracing_pipeline.h" "src/ray_tracing_pipeline.cpp" )

# Define GLM interface library manually
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${VCPKG_ROOT}/installed/x64-windows/include)

# Link Basalt library and GLM interface
target_link_libraries(VolumeRenderer PRIVATE Basalt glm)

# Set C++ standard
set_target_properties(VolumeRenderer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
