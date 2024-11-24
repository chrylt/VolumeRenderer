#pragma once

#include <vulkan/vulkan.h>
#include "device.h"

class RayTracingPipeline {
public:
    RayTracingPipeline(basalt::Device& device, VkDescriptorSetLayout descriptorSetLayout);
    ~RayTracingPipeline();

    VkPipeline getPipeline() const { return pipeline; }
    VkPipelineLayout getPipelineLayout() const { return pipelineLayout; }

private:
    void createShaderModules();
    void createPipeline();

    basalt::Device& device;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule raygenShaderModule = VK_NULL_HANDLE;
    VkShaderModule missShaderModule = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptorSetLayout;
};