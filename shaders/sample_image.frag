#version 450
layout (binding = 1) uniform sampler2D imgSampler;

layout (location = 0) out vec4 outColor;

void main() {
    outColor = texture(imgSampler, gl_FragCoord.xy / vec2(textureSize(imgSampler, 0)));
}