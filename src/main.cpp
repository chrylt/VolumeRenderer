#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <cstdlib>

#include <vulkan/vulkan_core.h>
#include <GLFW/glfw3.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <openvdb/openvdb.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreatePrimitives.h>

#include "buffer.h"
#include "command_buffer.h"
#include "command_pool.h"
#include "compute_pipeline.h"
#include "descriptor_pool.h"
#include "descriptor_set_layout.h"
#include "device.h"
#include "image.h"
#include "instance.h"
#include "graphics_pipeline.h"
#include "renderpass.h"
#include "surface.h"
#include "swapchain.h"
#include "sync_objects.h"
#include "utils.h"

// Constants for window dimensions
constexpr uint32_t WIDTH = 1024;
constexpr uint32_t HEIGHT = 1024;

// Maximum number of frames that can be processed concurrently
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

// Paths to compiled shader modules
const std::string LIGHT_GEN_PATH = "shaders/compiled_shaders/light_gen.comp.spv";
const std::string COMPUTE_SHADER_PATH = "shaders/compiled_shaders/compute_color.comp.spv";
const std::string VERT_SHADER_PATH = "shaders/compiled_shaders/fullscreen.vert.spv";
const std::string FRAG_SHADER_PATH = "shaders/compiled_shaders/sample_image.frag.spv";

struct RayLight {
    glm::vec3 positionFrom;
    glm::vec3 positionTo;
    float intensity;
};

struct LightCountBuffer
{
    uint32_t lightCounter;
    uint32_t debug;
};

struct UBO {
    alignas(4) uint32_t frameCount;
    alignas(8) glm::uvec2 framebufferDim;
    alignas(16) glm::vec3 cameraPos;
    alignas(4) float fov;
    alignas(4) float photonInitialIntensity;
    alignas(4) float scatteringProbability;
    alignas(4) float absorptionCoefficient;
    alignas(4) uint32_t maxLights;
    alignas(4) float rayMaxDistance;
    alignas(4) float rayMarchingStepSize;
    alignas(16) glm::vec3 lightSourceWorldPos;
};


class VolumeApp {
public:
    VolumeApp() {
        initWindow();
        initVulkan();
    }

    void run() {
        mainLoop();
        cleanup();
    }

private:
    // Window
    GLFWwindow* window;

    // Vulkan components
    std::unique_ptr<basalt::Instance> instance;
    std::unique_ptr<basalt::Surface> surface;
    std::unique_ptr<basalt::Device> device;
    std::unique_ptr<basalt::SwapChain> swapChain;
    std::unique_ptr<basalt::RenderPass> renderPass;
    std::unique_ptr<basalt::GraphicsPipeline> graphicsPipeline;
    std::unique_ptr<basalt::ComputePipeline> computePipeline;
    std::unique_ptr<basalt::ComputePipeline> lightGenPipeline;
    std::unique_ptr<basalt::CommandPool> commandPool;
    std::unique_ptr<basalt::SyncObjects> syncObjects;

    // Descriptor sets and layouts
    std::unique_ptr<basalt::DescriptorSetLayout> descriptorSetLayout;
    std::unique_ptr<basalt::DescriptorPool> descriptorPool;
    VkDescriptorSet descriptorSet;

    // Storage image
    std::unique_ptr<basalt::Image> storageImage;
    VkImageView storageImageView;
    VkSampler sampler;

    // Buffers
    std::unique_ptr<basalt::Buffer> nanoVDBBuffer;
    std::unique_ptr<basalt::Buffer> rayLightsBuffer;
    std::unique_ptr<basalt::Buffer> lightCounterBuffer;
    std::unique_ptr<basalt::Buffer> uniformBuffer;
    UBO uboData;

    // Pipelines
    VkPipelineLayout graphicsPipelineLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipelineLayout lightGenPipelineLayout;

    // Command buffers
    std::vector<std::unique_ptr<basalt::CommandBuffer>> commandBuffers;

    // Frame management
    size_t currentFrame = 0;
    bool framebufferResized = false;

    // Initialization methods
    void initWindow();
    void initVulkan();
    void createStorageImage();
    void createLightBuffers();
    void createDescriptorSetLayout();
    void createDescriptorPoolAndSet();
    void createLightGenPipeline();
    void createComputePipeline();
    void createGraphicsPipeline();
    void createCommandBuffers();
    void createSyncObjects();
    void createSampler();
    void createNanoVDBBuffer();
    void createUniformBuffer();

    // Rendering loop
    void mainLoop();

    // Frame rendering
    void drawFrame();

    // Swap chain recreation
    void recreateSwapChain();
    void updateDescriptorSet() const;
    void cleanupSwapChain();

    // Cleanup
    void cleanup();

    // Callback for framebuffer resize
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        const auto app = reinterpret_cast<VolumeApp*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
};

void VolumeApp::initWindow() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW!");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Volume Renderer", nullptr, nullptr);
    if (!window) {
        throw std::runtime_error("Failed to create GLFW window!");
    }

    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void VolumeApp::initVulkan() {
    instance = std::make_unique<basalt::Instance>();
    surface = std::make_unique<basalt::Surface>(*instance, window);
    device = std::make_unique<basalt::Device>(*instance, *surface);
    swapChain = std::make_unique<basalt::SwapChain>(*device, *surface, window);
    renderPass = std::make_unique<basalt::RenderPass>(*device, swapChain->getImageFormat());
    commandPool = std::make_unique<basalt::CommandPool>(*device, device->getGraphicsQueueFamilyIndex());

    createNanoVDBBuffer();
    createLightBuffers();
    createStorageImage();
    createSampler();
    createUniformBuffer();
    createDescriptorSetLayout();
    createDescriptorPoolAndSet();
    createLightGenPipeline();
    createComputePipeline();
    createGraphicsPipeline();
    createCommandBuffers();
    swapChain->createFramebuffers(*renderPass);
    createSyncObjects();
}

void VolumeApp::createStorageImage() {
    VkFormat imageFormat = basalt::utils::findSupportedFormat(*device,
        { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM }, // Candidate formats
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);

    storageImage = std::make_unique<basalt::Image>(
        *device,
        swapChain->getExtent().width,
        swapChain->getExtent().height,
        imageFormat,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = storageImage->getImage();
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = imageFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device->getDevice(), &viewInfo, nullptr, &storageImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view!");
    }

    // Transition image layout to GENERAL for storage image
    basalt::utils::transitionImageLayout(
        *device,
        *commandPool,
        device->getGraphicsQueue(),
        storageImage->getImage(),
        imageFormat,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL);
}

void VolumeApp::createLightBuffers()
{
    // Create a buffer for point lights
    VkDeviceSize maxLights = 100000;
    VkDeviceSize pointLightsBufferSize = sizeof(RayLight) * maxLights;
    rayLightsBuffer = std::make_unique<basalt::Buffer>(
        *device, pointLightsBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    // Buffer for the counter (uint)
    VkDeviceSize counterBufferSize = sizeof(LightCountBuffer);
    lightCounterBuffer = std::make_unique<basalt::Buffer>(
        *device, counterBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
}

void VolumeApp::createDescriptorSetLayout() {
    // Binding 0: Storage image for compute shader
    VkDescriptorSetLayoutBinding storageImageLayoutBinding{};
    storageImageLayoutBinding.binding = 0;
    storageImageLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    storageImageLayoutBinding.descriptorCount = 1;
    storageImageLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    storageImageLayoutBinding.pImmutableSamplers = nullptr;

    // Binding 1: Combined image sampler for fragment shader
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    // Binding 2: Storage buffer for NanoVDB grid
    VkDescriptorSetLayoutBinding storageBufferLayoutBinding{};
    storageBufferLayoutBinding.binding = 2;
    storageBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    storageBufferLayoutBinding.descriptorCount = 1;
    storageBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    storageBufferLayoutBinding.pImmutableSamplers = nullptr;

    // Binding 3: Point lights buffer
    VkDescriptorSetLayoutBinding pointLightsBinding{};
    pointLightsBinding.binding = 3;
    pointLightsBinding.descriptorCount = 1;
    pointLightsBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pointLightsBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Binding 4: Light counter buffer
    VkDescriptorSetLayoutBinding lightCounterBinding{};
    lightCounterBinding.binding = 4;
    lightCounterBinding.descriptorCount = 1;
    lightCounterBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    lightCounterBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Binding 5: Uniform buffer
    VkDescriptorSetLayoutBinding uniformBufferLayoutBinding{};
    uniformBufferLayoutBinding.binding = 5;
    uniformBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBufferLayoutBinding.descriptorCount = 1;
    uniformBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    uniformBufferLayoutBinding.pImmutableSamplers = nullptr;

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        storageImageLayoutBinding,
        samplerLayoutBinding,
        storageBufferLayoutBinding,
        pointLightsBinding,
        lightCounterBinding,
        uniformBufferLayoutBinding
    };

    descriptorSetLayout = std::make_unique<basalt::DescriptorSetLayout>(*device, bindings);
}

void VolumeApp::createDescriptorPoolAndSet() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}
    };

    descriptorPool = std::make_unique<basalt::DescriptorPool>(*device, 1, poolSizes);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool->getPool();
    allocInfo.descriptorSetCount = 1;
    const VkDescriptorSetLayout layout = descriptorSetLayout->getLayout();
    allocInfo.pSetLayouts = &layout;

    if (vkAllocateDescriptorSets(device->getDevice(), &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set!");
    }

    updateDescriptorSet();
}

void VolumeApp::createLightGenPipeline() {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    const VkDescriptorSetLayout layouts[] = { descriptorSetLayout->getLayout() };
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = layouts;

    if (vkCreatePipelineLayout(device->getDevice(), &pipelineLayoutInfo, nullptr, &lightGenPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create lightGen pipeline layout!");
    }

    lightGenPipeline = std::make_unique<basalt::ComputePipeline>(*device, LIGHT_GEN_PATH, lightGenPipelineLayout);
}

void VolumeApp::createComputePipeline() {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    const VkDescriptorSetLayout layouts[] = { descriptorSetLayout->getLayout() };
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = layouts;

    if (vkCreatePipelineLayout(device->getDevice(), &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline layout!");
    }

    computePipeline = std::make_unique<basalt::ComputePipeline>(*device, COMPUTE_SHADER_PATH, computePipelineLayout);
}

void VolumeApp::createSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(device->getDevice(), &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sampler!");
    }
}

void VolumeApp::createUniformBuffer()
{
    VkDeviceSize bufferSize = sizeof(UBO);
    uniformBuffer = std::make_unique<basalt::Buffer>(*device, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Set UBO values
    uboData.frameCount = 0;
    uboData.framebufferDim = glm::uvec2(swapChain->getExtent().width, swapChain->getExtent().height);
    uboData.cameraPos = glm::vec3(0.0, 20.0, -75.0);
    uboData.fov = 45.0f;
    uboData.photonInitialIntensity = 10.0f;
    uboData.scatteringProbability = 0.5f;
    uboData.absorptionCoefficient = 0.1f;
    uboData.maxLights = 1000;
    uboData.rayMaxDistance = 12000.0f;
    uboData.rayMarchingStepSize = 1.0f;
    uboData.lightSourceWorldPos = glm::vec3(-20.0, 15.0, -15.0);

    uniformBuffer->updateBuffer(*commandPool, &uboData, sizeof(UBO));
}


void VolumeApp::createGraphicsPipeline() {
    // Empty vertex input state
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr;

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    const VkDescriptorSetLayout layouts[] = { descriptorSetLayout->getLayout() };
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = layouts;

    if (vkCreatePipelineLayout(device->getDevice(), &pipelineLayoutInfo, nullptr, &graphicsPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics pipeline layout!");
    }

    graphicsPipeline = std::make_unique<basalt::GraphicsPipeline>(
        *device,
        *renderPass,
        *swapChain,
        VERT_SHADER_PATH,
        FRAG_SHADER_PATH,
        vertexInputInfo,
        descriptorSetLayout->getLayout());
}

void VolumeApp::createCommandBuffers() {
    // Allocate one command buffer per frame in flight
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        commandBuffers[i] = std::make_unique<basalt::CommandBuffer>(*device, *commandPool);
        // Do not pre-record commands here
    }
}

void VolumeApp::createSyncObjects() {
    syncObjects = std::make_unique<basalt::SyncObjects>(*device, MAX_FRAMES_IN_FLIGHT);
}

void VolumeApp::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }

    vkDeviceWaitIdle(device->getDevice());
}

void VolumeApp::drawFrame() {
    // Wait for the fence to be signaled before starting the frame
    syncObjects->waitForInFlightFence(currentFrame);

    uint32_t imageIndex;
    VkResult result = swapChain->acquireNextImage(*syncObjects, currentFrame, imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }

    // Reset the fence to the unsignaled state
    syncObjects->resetInFlightFence(currentFrame);

    // Record command buffer for the current frame
    basalt::CommandBuffer& cmdBuffer = *commandBuffers[currentFrame];

    // Update UBO:
    uboData.frameCount++;

    uniformBuffer->updateBuffer(*commandPool, &uboData, sizeof(UBO));

    // Begin recording
    cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    // Reset the light counter to zero before running light_gen
    vkCmdFillBuffer(
        *cmdBuffer.get(),
        lightCounterBuffer->getBuffer(),
        0,
        sizeof(LightCountBuffer),
        0 // fill with zero
    );

    // ---- First Compute Shader: light_gen ----
    vkCmdBindPipeline(*cmdBuffer.get(), VK_PIPELINE_BIND_POINT_COMPUTE, lightGenPipeline->getPipeline());
    vkCmdBindDescriptorSets(
        *cmdBuffer.get(),
        VK_PIPELINE_BIND_POINT_COMPUTE,
        lightGenPipelineLayout,
        0,
        1,
        &descriptorSet,
        0,
        nullptr
    );

    // Dispatch compute shader
    vkCmdDispatch(*cmdBuffer.get(), 1, 1, 1);   // just one block of local size 4x4

    // Insert memory barrier to ensure light data is written before the next compute shader
    VkMemoryBarrier memoryBarrier1{};
    memoryBarrier1.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier1.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        *cmdBuffer.get(),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &memoryBarrier1,
        0, nullptr,
        0, nullptr
    );

    // ---- Second Compute Shader: compute_gradient ----
    vkCmdBindPipeline(*cmdBuffer.get(), VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline->getPipeline());
    vkCmdBindDescriptorSets(
        *cmdBuffer.get(),
        VK_PIPELINE_BIND_POINT_COMPUTE,
        computePipelineLayout,
        0,
        1,
        &descriptorSet,
        0,
        nullptr
    );

    // Dispatch compute shader with workgroup size (16x16)
    const uint32_t groupCountX = (swapChain->getExtent().width + 15) / 16;
    const uint32_t groupCountY = (swapChain->getExtent().height + 15) / 16;
    vkCmdDispatch(*cmdBuffer.get(), groupCountX, groupCountY, 1);

    // Insert memory barrier to ensure compute shader writes are visible to the fragment shader
    VkImageMemoryBarrier imageBarrier{};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = storageImage->getImage();
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        *cmdBuffer.get(),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &imageBarrier
    );

    // ---- Graphics Pipeline: Render Pass ----
    VkClearValue clearColor = { {0.0f, 0.0f, 0.0f, 1.0f} };
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass->getRenderPass();
    renderPassInfo.framebuffer = swapChain->getFramebuffers()[imageIndex];
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = swapChain->getExtent();
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(*cmdBuffer.get(), &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Bind graphics pipeline
    vkCmdBindPipeline(*cmdBuffer.get(), VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline->getPipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(
        *cmdBuffer.get(),
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        graphicsPipelineLayout,
        0,
        1,
        &descriptorSet,
        0,
        nullptr
    );

    // Draw fullscreen triangle (no vertex buffer needed)
    vkCmdDraw(*cmdBuffer.get(), 3, 1, 0, 0);

    // End render pass
    vkCmdEndRenderPass(*cmdBuffer.get());

    // End recording
    cmdBuffer.end();

    // ---- Submit Command Buffer ----
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // Wait on the imageAvailableSemaphore
    VkSemaphore waitSemaphores[] = { syncObjects->getImageAvailableSemaphore(currentFrame) };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    // Specify the command buffer to submit
    VkCommandBuffer submitCmdBuffer = *cmdBuffer.get();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &submitCmdBuffer;

    // Signal the renderFinishedSemaphore when the command buffer finishes
    VkSemaphore signalSemaphores[] = { syncObjects->getRenderFinishedSemaphore(currentFrame) };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(device->getGraphicsQueue(), 1, &submitInfo, syncObjects->getInFlightFence(currentFrame)) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer!");
    }

    // ---- Present the Image ----
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    // Wait on the renderFinishedSemaphore
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    // Specify the swapchain and image to present
    VkSwapchainKHR swapChains[] = { swapChain->getSwapChain() };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(device->getPresentQueue(), &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swap chain image!");
    }

    // Advance to the next frame
    currentFrame = (currentFrame + 1) % syncObjects->getMaxFramesInFlight();
}

void VolumeApp::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &width, &height);
    }

    vkDeviceWaitIdle(device->getDevice());
    cleanupSwapChain();

    swapChain = std::make_unique<basalt::SwapChain>(*device, *surface, window);
    renderPass = std::make_unique<basalt::RenderPass>(*device, swapChain->getImageFormat());
    createStorageImage();
    updateDescriptorSet();
    createComputePipeline();
    createGraphicsPipeline();
    swapChain->createFramebuffers(*renderPass);
    createCommandBuffers();
}

void VolumeApp::updateDescriptorSet() const
{
    // Update descriptor set
    std::array<VkWriteDescriptorSet, 6> descriptorWrites{};

    // Binding 0: Storage image
    VkDescriptorImageInfo storageImageInfo{};
    storageImageInfo.imageView = storageImageView;
    storageImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pImageInfo = &storageImageInfo;

    // Binding 1: Combined image sampler
    VkDescriptorImageInfo samplerImageInfo{};
    samplerImageInfo.imageView = storageImageView;
    samplerImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    samplerImageInfo.sampler = sampler;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &samplerImageInfo;

    // Binding 2: Storage buffer
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = nanoVDBBuffer->getBuffer();
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = descriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &bufferInfo;

    // Binding 3: Light buffer
    VkDescriptorBufferInfo rayLightsBufferInfo{};
    rayLightsBufferInfo.buffer = rayLightsBuffer->getBuffer();
    rayLightsBufferInfo.offset = 0;
    rayLightsBufferInfo.range = VK_WHOLE_SIZE;

    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = descriptorSet;
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[3].pBufferInfo = &rayLightsBufferInfo;

    // Binding 4: Light count buffer
    VkDescriptorBufferInfo lightCounterBufferInfo{};
    lightCounterBufferInfo.buffer = lightCounterBuffer->getBuffer();
    lightCounterBufferInfo.offset = 0;
    lightCounterBufferInfo.range = VK_WHOLE_SIZE;

    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = descriptorSet;
    descriptorWrites[4].dstBinding = 4;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[4].pBufferInfo = &lightCounterBufferInfo;

    // Binding 5: Uniform buffer
    VkDescriptorBufferInfo uniformBufferInfo{};
    uniformBufferInfo.buffer = uniformBuffer->getBuffer();
    uniformBufferInfo.offset = 0;
    uniformBufferInfo.range = sizeof(UBO);

    descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[5].dstSet = descriptorSet;
    descriptorWrites[5].dstBinding = 5;
    descriptorWrites[5].descriptorCount = 1;
    descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[5].pBufferInfo = &uniformBufferInfo;

    vkUpdateDescriptorSets(device->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void VolumeApp::cleanupSwapChain() {
    vkDeviceWaitIdle(device->getDevice());

    // Destroy command buffers
    commandBuffers.clear();

    // Destroy storage image and image view
    if (storageImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device->getDevice(), storageImageView, nullptr);
        storageImageView = VK_NULL_HANDLE;
    }
    storageImage.reset();

    // Destroy pipelines first
    lightGenPipeline.reset();
    graphicsPipeline.reset();
    computePipeline.reset();

    // Then destroy pipeline layouts
    if (lightGenPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device->getDevice(), lightGenPipelineLayout, nullptr);
        lightGenPipelineLayout = VK_NULL_HANDLE;
    }
    if (graphicsPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device->getDevice(), graphicsPipelineLayout, nullptr);
        graphicsPipelineLayout = VK_NULL_HANDLE;
    }
    if (computePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device->getDevice(), computePipelineLayout, nullptr);
        computePipelineLayout = VK_NULL_HANDLE;
    }

    // Destroy framebuffers and render pass
    swapChain->cleanup();
    renderPass.reset();

    // Destroy swap chain
    swapChain.reset();
}

void VolumeApp::cleanup() {
    vkDeviceWaitIdle(device->getDevice());

    // Destroy storage image view if not already destroyed
    if (storageImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device->getDevice(), storageImageView, nullptr);
        storageImageView = VK_NULL_HANDLE;
    }

    // Destroy sampler
    if (sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device->getDevice(), sampler, nullptr);
        sampler = VK_NULL_HANDLE;
    }

    // Destroy pipelines and pipeline layouts
    // (Already handled in cleanupSwapChain())

    // Destroy the storage image
    storageImage.reset();

    // Destroy buffers
    nanoVDBBuffer.reset();
    rayLightsBuffer.reset();
    lightCounterBuffer.reset();
    uniformBuffer.reset();

    // Cleanup swap chain and other resources
    cleanupSwapChain();

    // Destroy descriptor pool and set layout
    descriptorPool.reset();
    descriptorSetLayout.reset();

    // Destroy command buffers
    commandBuffers.clear();

    // Destroy command pool and synchronization objects
    commandPool.reset();
    syncObjects.reset();

    // Destroy device, surface, and instance
    device.reset();
    surface.reset();
    instance.reset();

    // Destroy window and terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
}

void VolumeApp::createNanoVDBBuffer() {
    // Initialize OpenVDB
    openvdb::initialize();

    // Specify the VDB file to read
    const std::string filename = "../../../../resources/bunny_cloud.vdb";

    // Open the VDB file
    openvdb::io::File file(filename);
    try {
        file.open();
    }
    catch (const openvdb::IoError& e) {
        std::cerr << "Error opening file " << filename << ": " << e.what() << '\n';
        throw std::runtime_error("Failed to open VDB file.");
    }

    // Read the first FloatGrid from the file
    openvdb::GridBase::Ptr baseGrid_openvdb;
    openvdb::FloatGrid::Ptr grid_openvdb;

    for (openvdb::io::File::NameIterator nameIter = file.beginName();
        nameIter != file.endName(); ++nameIter) {
        baseGrid_openvdb = file.readGrid(nameIter.gridName());
        grid_openvdb = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid_openvdb);
        if (grid_openvdb) {
            std::cout << "Loaded grid: " << nameIter.gridName() << '\n';
            break;
        }
    }

    file.close();

    if (!grid_openvdb) {
        std::cerr << "No FloatGrid found in " << filename << '\n';
        openvdb::uninitialize();
        throw std::runtime_error("Failed to find FloatGrid in VDB file.");
    }

    // Create NanoVDB grid handle
    nanovdb::GridHandle<> handle = nanovdb::tools::createNanoGrid(*grid_openvdb);

    // Get the data and size
    const void* gridData = handle.data();
    const size_t gridSize = handle.size();

    // Create Vulkan buffer
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(gridSize);

    // Create a buffer with usage VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    nanoVDBBuffer = std::make_unique<basalt::Buffer>(
        *device,
        bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Create a staging buffer to transfer data to device local memory
    basalt::Buffer stagingBuffer(
        *device,
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Copy data to the staging buffer
    stagingBuffer.updateBuffer(*commandPool, gridData, bufferSize);

    // Copy from staging buffer to device local buffer
    basalt::utils::copyBuffer(*device, *commandPool, device->getGraphicsQueue(),
        stagingBuffer.getBuffer(), nanoVDBBuffer->getBuffer(), bufferSize);

    // Clean up OpenVDB
    openvdb::uninitialize();
}

int main() {
    try {
        VolumeApp app;
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::cin.get(); // wait before closing for debugging purposes
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
