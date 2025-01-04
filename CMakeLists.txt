cmake_minimum_required(VERSION 3.20)

# Enable NanoVDB and set OpenVDB to build as shared libraries before defining the project
set(USE_NANOVDB ON CACHE BOOL "Enable NanoVDB" FORCE)
set(OPENVDB_BUILD_CORE_STATIC OFF CACHE BOOL "Build OpenVDB core as shared library" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)

# Set CMake policies
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 NEW)
endif()

# Define the project and specify languages
project(VolumeRenderer LANGUAGES CXX CUDA)

# Add submodules
add_subdirectory(external/Basalt)
add_subdirectory(external/openvdb)
add_subdirectory(external/imgui)

# Find other packages (excluding GLM)
find_package(glfw3 REQUIRED)
find_package(assimp REQUIRED)
find_package(Vulkan REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TBB CONFIG REQUIRED)

# Add Shader directory
add_subdirectory(shaders)

# Add CPU test
add_subdirectory(CPU_test)

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
)

# Create executable
add_executable(VolumeRenderer ${SOURCES})

# Fix OpenVDB NanoVDB compatibility issues
target_compile_definitions(VolumeRenderer PRIVATE NANOVDB_USE_OPENVDB)

# Define GLM interface library manually
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${VCPKG_ROOT}/installed/x64-windows/include)

# Link Basalt library, GLM interface, OpenVDB, and Vulkan
target_link_libraries(VolumeRenderer PRIVATE Basalt glm openvdb Vulkan::Vulkan imgui)

# Set C++ standard
set_target_properties(VolumeRenderer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Add compiler flags for Visual Studio to handle large templates and object files
if(MSVC)
    target_compile_options(VolumeRenderer PRIVATE /Zm200 /bigobj)
endif()

# Define macros based on configuration
target_compile_definitions(VolumeRenderer PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
)

# Set optimization flags for Release configuration
target_compile_options(VolumeRenderer PRIVATE
    $<$<CONFIG:Release>:/O2>
)

# Organize output directories
set_target_properties(VolumeRenderer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)

# Add post-build commands to copy runtime DLLs and set working directory based on configuration
if(MSVC)
    add_custom_command(TARGET VolumeRenderer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:VolumeRenderer>
            $<TARGET_FILE_DIR:VolumeRenderer>
        COMMAND_EXPAND_LISTS
    )
    
    # Set the debugger working directory based on the build configuration
    set_property(TARGET VolumeRenderer PROPERTY 
        VS_DEBUGGER_WORKING_DIRECTORY "$<IF:$<CONFIG:Debug>,${CMAKE_SOURCE_DIR}/out/build/x64-debug,${CMAKE_SOURCE_DIR}/out/build/x64-release>"
    )
endif()
