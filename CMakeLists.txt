cmake_minimum_required(VERSION 3.10)
project(VolumeRenderer LANGUAGES CXX CUDA)

# Add submodules
add_subdirectory(external/Basalt)
add_subdirectory(external/openvdb)

# Find other packages (excluding GLM)
find_package(glfw3 REQUIRED)
find_package(stb REQUIRED)
find_package(assimp REQUIRED)
find_package(Vulkan REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TBB CONFIG REQUIRED)

# Add Shader directory
add_subdirectory(shaders)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/Basalt/include
    ${CMAKE_SOURCE_DIR}/external/openvdb/nanovdb
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    include/ray_tracing_pipeline.h
    src/ray_tracing_pipeline.cpp
)

# Create executable
add_executable(VolumeRenderer ${SOURCES})

# Fix openvdb nanovdb compatibility issues
target_compile_definitions(VolumeRenderer PRIVATE NANOVDB_USE_OPENVDB)

# Define GLM interface library manually
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${VCPKG_ROOT}/installed/x64-windows/include)

# Link Basalt library, GLM interface, OpenVDB, and Vulkan
target_link_libraries(VolumeRenderer PRIVATE Basalt glm openvdb Vulkan::Vulkan)

# Set C++ standard
set_target_properties(VolumeRenderer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Add subexecutables
#add_subdirectory(tests)