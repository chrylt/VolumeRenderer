add_executable(HelloTriangle main.cpp)

target_link_libraries(HelloTriangle PRIVATE Basalt)

# Ensure shaders are compiled before building the example
add_dependencies(HelloTriangle Shaders)

# Include directories
target_include_directories(HelloTriangle PRIVATE ${Vulkan_INCLUDE_DIRS} ${GLM_INCLUDE_DIRS})

# Copy compiled shaders to the output directory
add_custom_command(TARGET HelloTriangle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_BINARY_DIR}/shaders/compiled_shaders $<TARGET_FILE_DIR:HelloTriangle>/shaders/compiled_shaders
)
