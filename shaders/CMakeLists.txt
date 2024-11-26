# Directory for compiled SPIR-V shaders
set(SPIRV_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/compiled_shaders")

# Ensure the output directory exists
file(MAKE_DIRECTORY ${SPIRV_OUTPUT_DIR})

# Clear any existing SPIR-V files in the list (to prevent duplicates)
set(SPIRV_FILES)

# Define the include directory for NanoVDB GLSL headers
#set(NANOVDB_GLSL_INCLUDE_DIR "path/to/nanovdb/glsl")
#${VCPKG_ROOT}/installed/x64-windows/include

# Function to compile GLSL shaders to SPIR-V
function(compile_shaders shader_dir output_dir)
    # Find all GLSL files in the shader directory
    file(GLOB_RECURSE SHADER_FILES
        "${shader_dir}/*.vert"
        "${shader_dir}/*.frag"
        "${shader_dir}/*.comp"
        "${shader_dir}/*.rgen"
        "${shader_dir}/*.rmiss"
        "${shader_dir}/*.rchit"
        "${shader_dir}/*.rahit"
        "${shader_dir}/*.rint"
        "${shader_dir}/*.rcall"
    )

    # Loop over each shader file and compile to SPIR-V
    foreach(SHADER_FILE ${SHADER_FILES})
        get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
        get_filename_component(SHADER_EXT ${SHADER_FILE} EXT)
        string(SUBSTRING ${SHADER_EXT} 1 -1 SHADER_STAGE)

        # Map file extensions to shader stages
        if(${SHADER_STAGE} STREQUAL "vert")
            set(SHADER_STAGE_FLAG "vert")
        elseif(${SHADER_STAGE} STREQUAL "frag")
            set(SHADER_STAGE_FLAG "frag")
        elseif(${SHADER_STAGE} STREQUAL "comp")
            set(SHADER_STAGE_FLAG "comp")
        elseif(${SHADER_STAGE} STREQUAL "rgen")
            set(SHADER_STAGE_FLAG "rgen")
        elseif(${SHADER_STAGE} STREQUAL "rmiss")
            set(SHADER_STAGE_FLAG "rmiss")
        elseif(${SHADER_STAGE} STREQUAL "rchit")
            set(SHADER_STAGE_FLAG "rchit")
        elseif(${SHADER_STAGE} STREQUAL "rahit")
            set(SHADER_STAGE_FLAG "rahit")
        elseif(${SHADER_STAGE} STREQUAL "rint")
            set(SHADER_STAGE_FLAG "rint")
        elseif(${SHADER_STAGE} STREQUAL "rcall")
            set(SHADER_STAGE_FLAG "rcall")
        else()
            message(FATAL_ERROR "Unknown shader stage for file ${SHADER_FILE}")
        endif()

        # Define output path for the SPIR-V file
        set(SPIRV_OUTPUT "${output_dir}/${SHADER_NAME}.${SHADER_STAGE}.spv")

        message(STATUS "Compiling ${SHADER_FILE} to ${SPIRV_OUTPUT}")

        # Add a custom command to compile GLSL to SPIR-V
        add_custom_command(
            OUTPUT ${SPIRV_OUTPUT}
            COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
                    -V --target-env vulkan1.2
                    -S ${SHADER_STAGE_FLAG}
                    #-I${NANOVDB_GLSL_INCLUDE_DIR}
                    -o ${SPIRV_OUTPUT}
                    ${SHADER_FILE}
            DEPENDS ${SHADER_FILE}
            COMMENT "Compiling ${SHADER_FILE} to SPIR-V"
        )

        list(APPEND SPIRV_FILES ${SPIRV_OUTPUT})
    endforeach()

    set(SPIRV_FILES ${SPIRV_FILES} PARENT_SCOPE)
endfunction()



# Call the function to compile shaders
compile_shaders(${CMAKE_CURRENT_SOURCE_DIR} ${SPIRV_OUTPUT_DIR})

# Create a custom target for shader compilation
add_custom_target(Shaders ALL DEPENDS ${SPIRV_FILES})
