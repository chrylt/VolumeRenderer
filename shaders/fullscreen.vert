#version 450
#extension GL_KHR_vulkan_glsl : enable

void main() {
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
