# Directory for compiled SPIR-V shaders
set(SPIRV_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/compiled_shaders")

# Ensure the output directory exists
file(MAKE_DIRECTORY ${SPIRV_OUTPUT_DIR})

# Clear any existing SPIR-V files in the list (to prevent duplicates)
set(SPIRV_FILES)

# Function to compile GLSL shaders to SPIR-V
function(compile_shaders shader_dir output_dir)
    # Find all GLSL files in the shader directory
    file(GLOB_RECURSE SHADER_FILES "${shader_dir}/*.vert" "${shader_dir}/*.frag" "${shader_dir}/*.comp")

    # Debugging: Print out the found shader files
    message(STATUS "Found shader files:")
    foreach(SHADER_FILE ${SHADER_FILES})
        message(STATUS "  ${SHADER_FILE}")
    endforeach()

    # Loop over each shader file and compile to SPIR-V
    foreach(SHADER_FILE ${SHADER_FILES})
        get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)  # Filename without extension
        get_filename_component(SHADER_EXT ${SHADER_FILE} EXT)

        # Define output path for the SPIR-V file
        set(SPIRV_OUTPUT "${output_dir}/${SHADER_NAME}${SHADER_EXT}.spv")

        # Debugging: Print output path for SPIR-V file
        message(STATUS "Compiling ${SHADER_FILE} to ${SPIRV_OUTPUT}")

        # Add a custom command to compile GLSL to SPIR-V
        add_custom_command(
            OUTPUT ${SPIRV_OUTPUT}
            COMMAND glslangValidator -V ${SHADER_FILE} -o ${SPIRV_OUTPUT}
            DEPENDS ${SHADER_FILE}  # Recompile if the source file changes
            COMMENT "Compiling ${SHADER_FILE} to SPIR-V"
        )

        # Collect SPIR-V files for the main target dependency
        list(APPEND SPIRV_FILES ${SPIRV_OUTPUT})
    endforeach()

    # Set SPIRV_FILES as a cache variable to make it accessible at the root level
    set(SPIRV_FILES ${SPIRV_FILES} PARENT_SCOPE)
endfunction()

# Call the function to compile shaders
compile_shaders(${CMAKE_CURRENT_SOURCE_DIR} ${SPIRV_OUTPUT_DIR})

# Create a custom target for shader compilation
add_custom_target(Shaders ALL DEPENDS ${SPIRV_FILES})
