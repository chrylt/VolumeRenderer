#include "ray_tracing_pipeline.h"
#include "utils.h"
#include <vector>
#include <stdexcept>

RayTracingPipeline::RayTracingPipeline(basalt::Device& device, VkDescriptorSetLayout descriptorSetLayout)
    : device(device), descriptorSetLayout(descriptorSetLayout) {
    createShaderModules();
    createPipeline();
}

RayTracingPipeline::~RayTracingPipeline() {
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device.getDevice(), pipeline, nullptr);
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device.getDevice(), pipelineLayout, nullptr);
    }

    if (raygenShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device.getDevice(), raygenShaderModule, nullptr);
    }
    if (missShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device.getDevice(), missShaderModule, nullptr);
    }
}

// Helper function to create shader modules
static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    if (code.empty()) {
        throw std::runtime_error("Shader code is empty!");
    }

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    if (code.size() % 4 != 0) {
        throw std::runtime_error("Shader code size must be a multiple of 4 bytes");
    }
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }
    return shaderModule;
}

void RayTracingPipeline::createShaderModules() {
    // Load shader code
    auto raygenCode = basalt::utils::readFile("shaders/compiled_shaders/raygen.rgen.spv");
    auto missCode = basalt::utils::readFile("shaders/compiled_shaders/miss.rmiss.spv");
    // Load other shaders as needed

    // Create shader modules
    raygenShaderModule = createShaderModule(device.getDevice(), raygenCode);
    missShaderModule = createShaderModule(device.getDevice(), missCode);
    // Create other shader modules if needed
}

void RayTracingPipeline::createPipeline() {
    /*
    // Set up shader stages
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

    // Raygen shader
    VkPipelineShaderStageCreateInfo raygenShaderStage{};
    raygenShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    raygenShaderStage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    raygenShaderStage.module = raygenShaderModule;
    raygenShaderStage.pName = "main";
    shaderStages.push_back(raygenShaderStage);

    // Miss shader
    VkPipelineShaderStageCreateInfo missShaderStage{};
    missShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    missShaderStage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    missShaderStage.module = missShaderModule;
    missShaderStage.pName = "main";
    shaderStages.push_back(missShaderStage);

    // Set up shader groups
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;

    // Raygen group
    VkRayTracingShaderGroupCreateInfoKHR raygenGroup{};
    raygenGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    raygenGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    raygenGroup.generalShader = 0;  // Index of raygen shader
    raygenGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
    raygenGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    raygenGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(raygenGroup);

    // Miss group
    VkRayTracingShaderGroupCreateInfoKHR missGroup{};
    missGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    missGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    missGroup.generalShader = 1;  // Index of miss shader
    missGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
    missGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    missGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(missGroup);

    // Create pipeline layout with descriptor set layouts
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    // add push constants here

    if (vkCreatePipelineLayout(device.getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }

    // Get ray tracing pipeline properties
    const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rayTracingPipelineProperties = device.getRayTracingProperties();

    // Create ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.groupCount = static_cast<uint32_t>(shaderGroups.size());
    pipelineInfo.pGroups = shaderGroups.data();
    pipelineInfo.maxPipelineRayRecursionDepth = 1;
    pipelineInfo.layout = pipelineLayout;

    // Create the ray tracing pipeline
    if (vkCreateRayTracingPipelinesKHR(device.getDevice(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ray tracing pipeline!");
    }
    */
}




