#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <cstdlib>

#include <vulkan/vulkan_core.h>
#include <GLFW/glfw3.h>

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
constexpr uint32_t WIDTH = 256;
constexpr uint32_t HEIGHT = 256;

// Maximum number of frames that can be processed concurrently
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

// Paths to compiled shader modules
const std::string COMPUTE_SHADER_PATH = "shaders/compiled_shaders/compute_gradient.comp.spv";
const std::string VERT_SHADER_PATH = "shaders/compiled_shaders/fullscreen.vert.spv";
const std::string FRAG_SHADER_PATH = "shaders/compiled_shaders/sample_image.frag.spv";

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

    // NanoVDB
    std::unique_ptr<basalt::Buffer> nanoVDBBuffer;

    // Pipelines
    VkPipelineLayout graphicsPipelineLayout;
    VkPipelineLayout computePipelineLayout;

    // Command buffers
    std::vector<std::unique_ptr<basalt::CommandBuffer>> commandBuffers;

    // Frame management
    size_t currentFrame = 0;
    bool framebufferResized = false;

    // Initialization methods
    void initWindow();
    void initVulkan();
    void createStorageImage();
    void createDescriptorSetLayout();
    void createDescriptorPoolAndSet();
    void createComputePipeline();
    void createGraphicsPipeline();
    void createCommandBuffers();
    void createSyncObjects();
    void createSampler();
    void createNanoVDBBuffer();

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
    createStorageImage();
    createSampler();
    createDescriptorSetLayout();
    createDescriptorPoolAndSet();
    createComputePipeline();
    createGraphicsPipeline();
    swapChain->createFramebuffers(*renderPass);
    createCommandBuffers();
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
    viewInfo.subresourceRange.levelCount = 1;
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

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        storageImageLayoutBinding,
        samplerLayoutBinding,
        storageBufferLayoutBinding
    };

    descriptorSetLayout = std::make_unique<basalt::DescriptorSetLayout>(*device, bindings);
}


void VolumeApp::createDescriptorPoolAndSet() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 }
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

    // Update descriptor set
    std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

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

    vkUpdateDescriptorSets(device->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
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

    if (vkCreateSampler(device->getDevice(), &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sampler!");
    }
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
    commandBuffers.resize(swapChain->getFramebuffers().size());

    for (size_t i = 0; i < commandBuffers.size(); ++i) {
        commandBuffers[i] = std::make_unique<basalt::CommandBuffer>(*device, *commandPool);

        commandBuffers[i]->begin(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

        // Bind compute pipeline and dispatch
        vkCmdBindPipeline(*commandBuffers[i]->get(), VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline->getPipeline());
        vkCmdBindDescriptorSets(
            *commandBuffers[i]->get(),
            VK_PIPELINE_BIND_POINT_COMPUTE,
            computePipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            nullptr);

        // Dispatch compute shader with workgroup size (16x16)
        const uint32_t groupCountX = (swapChain->getExtent().width + 15) / 16;
        const uint32_t groupCountY = (swapChain->getExtent().height + 15) / 16;
        vkCmdDispatch(*commandBuffers[i]->get(), groupCountX, groupCountY, 1);

        // Memory barrier to ensure compute shader writes are visible to the fragment shader
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = storageImage->getImage();
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(
            *commandBuffers[i]->get(),
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &barrier);

        // Begin render pass
        const VkClearValue clearColor = { {0.0f, 0.0f, 0.0f, 1.0f} };
        commandBuffers[i]->beginRenderPass(renderPass->getRenderPass(), swapChain->getFramebuffers()[i], swapChain->getExtent(), clearColor);

        // Bind graphics pipeline
        commandBuffers[i]->bindPipeline(graphicsPipeline->getPipeline());

        // Bind descriptor sets
        vkCmdBindDescriptorSets(
            *commandBuffers[i]->get(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            graphicsPipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            nullptr);

        // Draw fullscreen triangle (no vertex buffer needed)
        vkCmdDraw(*commandBuffers[i]->get(), 3, 1, 0, 0);

        commandBuffers[i]->endRenderPass();
        commandBuffers[i]->end();
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

    syncObjects->resetInFlightFence(currentFrame);

    const VkSemaphore waitSemaphores[] = { syncObjects->getImageAvailableSemaphore(currentFrame) };
    constexpr VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    const VkSemaphore signalSemaphores[] = { syncObjects->getRenderFinishedSemaphore(currentFrame) };

    if (device->submitCommandBuffers(
        commandBuffers[imageIndex]->get(), 1,
        waitSemaphores, 1,
        waitStages,
        signalSemaphores, 1,
        syncObjects->getInFlightFence(currentFrame)) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer!");
    }

    result = swapChain->presentImage(*syncObjects, currentFrame, imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swap chain image!");
    }

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
    std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

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

    // Destroy pipelines and pipeline layouts
    if (graphicsPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device->getDevice(), graphicsPipelineLayout, nullptr);
        graphicsPipelineLayout = VK_NULL_HANDLE;
    }
    if (computePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device->getDevice(), computePipelineLayout, nullptr);
        computePipelineLayout = VK_NULL_HANDLE;
    }
    graphicsPipeline.reset();
    computePipeline.reset();

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

    // Destroy pipeline layouts
    if (graphicsPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device->getDevice(), graphicsPipelineLayout, nullptr);
        graphicsPipelineLayout = VK_NULL_HANDLE;
    }
    if (computePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device->getDevice(), computePipelineLayout, nullptr);
        computePipelineLayout = VK_NULL_HANDLE;
    }

    // Destroy the storage image
    storageImage.reset();

    // Destroy nanoVDB buffer
    nanoVDBBuffer.reset();

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
    const basalt::Buffer stagingBuffer(
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
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
