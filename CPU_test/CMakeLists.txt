add_executable(CPUTest main.cpp)

# Fix OpenVDB NanoVDB compatibility issues
target_compile_definitions(CPUTest PRIVATE NANOVDB_USE_OPENVDB)

target_link_libraries(CPUTest PRIVATE Basalt glm openvdb)

# Ensure shaders are compiled before building the example
add_dependencies(CPUTest Shaders)

# Include directories
target_include_directories(CPUTest PRIVATE 
${GLM_INCLUDE_DIRS}  
${CMAKE_SOURCE_DIR}/external/openvdb/nanovdb 
${CUDAToolkit_INCLUDE_DIRS} 
)


# Copy compiled shaders to the output directory
add_custom_command(TARGET CPUTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_BINARY_DIR}/shaders/compiled_shaders $<TARGET_FILE_DIR:CPUTest>/shaders/compiled_shaders
)

# Add post-build commands to copy runtime DLLs
if(MSVC)
    add_custom_command(TARGET CPUTest POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:CPUTest>
            $<TARGET_FILE_DIR:CPUTest>
        COMMAND_EXPAND_LISTS
    )
    # Ensure that the runtime dependencies are copied correctly
    set_property(TARGET CPUTest PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/out/build/x64-debug")
endif()