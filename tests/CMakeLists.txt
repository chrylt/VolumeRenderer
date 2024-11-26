
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)


add_executable(HelloNanoVDB main.cpp nanovdb.cu)

set_target_properties(HelloNanoVDB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(HelloNanoVDB PROPERTIES CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 60 70 75)

target_link_libraries(HelloNanoVDB PRIVATE Basalt CUDA::cudart CUDA::cublas)

# Include directories
target_include_directories(HelloNanoVDB PRIVATE ${Vulkan_INCLUDE_DIRS} ${GLM_INCLUDE_DIRS})

target_compile_definitions(HelloNanoVDB PRIVATE NANOVDB_USE_OPENVDB)

# Copy compiled shaders to the output directory
add_custom_command(TARGET HelloNanoVDB POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
)
