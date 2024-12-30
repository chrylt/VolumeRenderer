layout(binding = 0, rgba8) uniform image2D img;

layout(binding = 1) uniform sampler2D imgSampler; // only used in last full-screen pass

// NanoVDB buffer
layout(binding = 2) readonly buffer NanoVDBBuffer {
    uint pnanovdb_buf_data[];
};

layout(std430, binding = 3) buffer PointLightBuffer {
    RayLight lights[];
};

layout(std430, binding = 4) buffer LightCounterBuffer {
    uint lightCount;
    uint debug;
};

layout(std140, binding = 5) uniform UBO {
    uint frameCount;
    uvec2 framebufferDim;
    vec3 cameraPos;
    float fov;
    float photonInitialIntensity;
    float scatteringProbability;
    float absorptionCoefficient;
    uint maxLights;
    float rayMaxDistance;
    float rayMarchingStepSize;
    vec3 lightSourceWorldPos;
    float beamRadius;
    float lightRayStepSize;
    float radiusFalloff;
};