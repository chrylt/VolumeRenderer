#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <array>

#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

#include "buffer.h"
#include "command_buffer.h"
#include "command_pool.h"
#include "device.h"
#include "instance.h"
#include "shader_module.h"
#include "surface.h"
#include "swapchain.h"
#include "sync_objects.h"
#include "utils.h"

#include "openvdb/openvdb.h"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/HostBuffer.h"
//#include "nanovdb/GridHandle.h"
#include "nanovdb/tools/CreateNanoGrid.h"

// Constants for window dimensions
constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

// Maximum number of frames that can be processed concurrently
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

// Paths to compiled shader modules
const std::string RAYGEN_SHADER_PATH = "shaders/compiled_shaders/raygen.rgen.spv";
const std::string MISS_SHADER_PATH = "shaders/compiled_shaders/miss.rmiss.spv";
const std::string TEST_VOLUME_PATH = "../../../resources/bunny_cloud.vdb";

// Function to align sizes
inline uint32_t alignedSize(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

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

    // Vulkan components managed by smart pointers for automatic cleanup
    std::unique_ptr<basalt::Instance> instance;
    std::unique_ptr<basalt::Surface> surface;
    std::unique_ptr<basalt::Device> device;
    std::unique_ptr<basalt::SwapChain> swapChain;
    std::unique_ptr<basalt::CommandPool> commandPool;
    std::unique_ptr<basalt::SyncObjects> syncObjects;

    // Ray tracing components
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline rayTracingPipeline;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    // Storage image for ray tracing output
    VkImage storageImage;
    VkDeviceMemory storageImageMemory;
    VkImageView storageImageView;

    // Shader Binding Table
    VkBuffer shaderBindingTableBuffer;
    VkDeviceMemory shaderBindingTableMemory;
    VkDeviceSize shaderBindingTableSize;

    // Volume data
    void* nanoVDBData = nullptr;
    size_t nanoVDBDataSize = 0;
    std::unique_ptr<basalt::Buffer> volumeBuffer;

    // Command buffers
    std::vector<VkCommandBuffer> commandBuffers;

    // Frame management
    size_t currentFrame = 0;
    bool framebufferResized = false;

    // Initialization methods
    void initWindow();
    void initVulkan();
    void createCommandBuffers();
    void createSyncObjects();
    void loadVolumeData();
    void createVolumeBuffer();
    void createDescriptorSetLayout();
    void createDescriptorSets();
    void createStorageImage();
    void createRayTracingPipeline();
    void createShaderBindingTable();
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

    // Rendering loop
    void mainLoop();

    // Frame rendering
    void drawFrame();

    // Swap chain recreation handled within SwapChain class
    void recreateSwapChain();
    void cleanup();

    // Callback for framebuffer resize
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<VolumeApp*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
};

void VolumeApp::initWindow() {
    // Initialize GLFW
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW!");
    }

    // No default API (Vulkan)
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    // Create GLFW window
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Volume Rendering", nullptr, nullptr);
    if (!window) {
        throw std::runtime_error("Failed to create GLFW window!");
    }

    // Set the user pointer to this class instance for callback access
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void VolumeApp::initVulkan() {
    // Create Vulkan instance
    instance = std::make_unique<basalt::Instance>();

    // Create surface associated with the window
    surface = std::make_unique<basalt::Surface>(*instance, window);

    // Create logical device and retrieve queues
    device = std::make_unique<basalt::Device>(*instance, *surface);

    // Create swap chain
    swapChain = std::make_unique<basalt::SwapChain>(*device, *surface, window);

    // Create command pool
    commandPool = std::make_unique<basalt::CommandPool>(*device, device->getGraphicsQueueFamilyIndex());

    // Load volume data
    loadVolumeData();

    // Create volume buffer
    createVolumeBuffer();

    // Create storage image
    createStorageImage();

    // Create descriptor set layout
    createDescriptorSetLayout();

    // Create ray tracing pipeline
    createRayTracingPipeline();

    // Create descriptor sets
    createDescriptorSets();

    // Create shader binding table
    createShaderBindingTable();

    // Allocate and record command buffers
    createCommandBuffers();

    // Create synchronization objects
    createSyncObjects();
}


void VolumeApp::loadVolumeData() {
    
    // Initialize OpenVDB
    openvdb::initialize();

    // Open the VDB file
    openvdb::io::File file(TEST_VOLUME_PATH);
    file.open();

    // Retrieve the first grid name from the file
    auto nameIter = file.beginName();
    if (nameIter == file.endName()) {
        throw std::runtime_error("No grids found in the VDB file!");
    }

    // Read the grid using its name
    auto baseGrid = file.readGrid(nameIter.gridName());
    file.close();

    // Cast to FloatGrid (or the appropriate grid type)
    auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
    if (!floatGrid) {
        throw std::runtime_error("Failed to cast grid to FloatGrid!");
    }

    // Convert OpenVDB grid to NanoVDB grid
    auto nanoGridHandle = nanovdb::tools::createNanoGrid(*floatGrid);

    // Store NanoVDB data
    nanoVDBDataSize = nanoGridHandle.size();
    nanoVDBData = malloc(nanoVDBDataSize);
    if (!nanoVDBData) {
        throw std::runtime_error("Failed to allocate memory for NanoVDB data!");
    }
    memcpy(nanoVDBData, nanoGridHandle.data(), nanoVDBDataSize);
}

void VolumeApp::createVolumeBuffer() {
    VkDeviceSize bufferSize = nanoVDBDataSize;

    // Create staging buffer
    basalt::Buffer stagingBuffer(*device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    stagingBuffer.updateBuffer(*commandPool, nanoVDBData, bufferSize);

    // Create device-local buffer
    volumeBuffer = std::make_unique<basalt::Buffer>(*device, bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Copy data from staging buffer to device-local buffer
    basalt::utils::copyBuffer(*device, *commandPool, device->getGraphicsQueue(),
        stagingBuffer.getBuffer(), volumeBuffer->getBuffer(), bufferSize);
}

void VolumeApp::createStorageImage() {
    VkExtent3D imageExtent = {
        swapChain->getExtent().width,
        swapChain->getExtent().height,
        1
    };

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = swapChain->getImageFormat();
    imageCreateInfo.extent = imageExtent;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device->getDevice(), &imageCreateInfo, nullptr, &storageImage) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create storage image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device->getDevice(), storageImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = device->findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device->getDevice(), &allocInfo, nullptr, &storageImageMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate storage image memory!");
    }

    vkBindImageMemory(device->getDevice(), storageImage, storageImageMemory, 0);

    // Create image view
    VkImageViewCreateInfo viewCreateInfo{};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.image = storageImage;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = swapChain->getImageFormat();
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCreateInfo.subresourceRange.baseMipLevel = 0;
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.subresourceRange.baseArrayLayer = 0;
    viewCreateInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device->getDevice(), &viewCreateInfo, nullptr, &storageImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create storage image view!");
    }

    // Transition image layout to VK_IMAGE_LAYOUT_GENERAL
    basalt::utils::transitionImageLayout(
        *device, *commandPool, device->getGraphicsQueue(),
        storageImage, swapChain->getImageFormat(),
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL
    );
}

void VolumeApp::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding volumeBinding{};
    volumeBinding.binding = 0;
    volumeBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    volumeBinding.descriptorCount = 1;
    volumeBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    volumeBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding storageImageBinding{};
    storageImageBinding.binding = 1;
    storageImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    storageImageBinding.descriptorCount = 1;
    storageImageBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    storageImageBinding.pImmutableSamplers = nullptr;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = { volumeBinding, storageImageBinding };
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device->getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }
}

void VolumeApp::createDescriptorSets() {
    // Create descriptor pool
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(device->getDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool!");
    }

    // Allocate descriptor sets
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(device->getDevice(), &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set!");
    }

    // Update descriptor sets
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = volumeBuffer->getBuffer();
    bufferInfo.offset = 0;
    bufferInfo.range = nanoVDBDataSize;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = storageImageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfo;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(device->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void VolumeApp::createRayTracingPipeline() {
    /*
    // Load shader modules
    basalt::ShaderModule raygenShaderModule(*device, RAYGEN_SHADER_PATH);
    basalt::ShaderModule missShaderModule(*device, MISS_SHADER_PATH);

    VkShaderModule raygenShader = raygenShaderModule.getShaderModule();
    VkShaderModule missShader = missShaderModule.getShaderModule();

    // Set up shader stages
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

    // Raygen shader
    VkPipelineShaderStageCreateInfo raygenShaderStageInfo{};
    raygenShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    raygenShaderStageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    raygenShaderStageInfo.module = raygenShader;
    raygenShaderStageInfo.pName = "main";
    shaderStages.push_back(raygenShaderStageInfo);

    // Miss shader
    VkPipelineShaderStageCreateInfo missShaderStageInfo{};
    missShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    missShaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    missShaderStageInfo.module = missShader;
    missShaderStageInfo.pName = "main";
    shaderStages.push_back(missShaderStageInfo);

    // Set up shader groups
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;

    // Raygen group
    VkRayTracingShaderGroupCreateInfoKHR raygenGroup{};
    raygenGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    raygenGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    raygenGroup.generalShader = 0; // Index of raygen shader
    raygenGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
    raygenGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    raygenGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(raygenGroup);

    // Miss group
    VkRayTracingShaderGroupCreateInfoKHR missGroup{};
    missGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    missGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    missGroup.generalShader = 1; // Index of miss shader
    missGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
    missGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    missGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(missGroup);

    // Create pipeline layout with descriptor set layouts
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    // Add push constants here if needed

    if (vkCreatePipelineLayout(device->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }

    // Create ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.groupCount = static_cast<uint32_t>(shaderGroups.size());
    pipelineInfo.pGroups = shaderGroups.data();
    pipelineInfo.maxPipelineRayRecursionDepth = 1;
    pipelineInfo.layout = pipelineLayout;

    if (vkCreateRayTracingPipelinesKHR(device->getDevice(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &rayTracingPipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ray tracing pipeline!");
    }
    */
}

void VolumeApp::createShaderBindingTable() {
    /*
    const uint32_t groupCount = 2; // Number of shader groups
    const uint32_t handleSize = device->getRayTracingProperties().shaderGroupHandleSize;

    // Compute SBT size
    const uint32_t handleSizeAligned = alignedSize(handleSize, device->getRayTracingProperties().shaderGroupHandleAlignment);
    shaderBindingTableSize = groupCount * handleSizeAligned;

    // Allocate buffer
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = shaderBindingTableSize;
    bufferInfo.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device->getDevice(), &bufferInfo, nullptr, &shaderBindingTableBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader binding table buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device->getDevice(), shaderBindingTableBuffer, &memRequirements);

    VkMemoryAllocateFlagsInfo allocFlagsInfo{};
    allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = device->findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    allocInfo.pNext = &allocFlagsInfo;

    if (vkAllocateMemory(device->getDevice(), &allocInfo, nullptr, &shaderBindingTableMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate shader binding table memory!");
    }

    vkBindBufferMemory(device->getDevice(), shaderBindingTableBuffer, shaderBindingTableMemory, 0);

    // Get shader group handles
    std::vector<uint8_t> shaderHandleStorage(shaderBindingTableSize);
    if (vkGetRayTracingShaderGroupHandlesKHR(device->getDevice(), rayTracingPipeline, 0, groupCount, shaderBindingTableSize, shaderHandleStorage.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to get shader group handles!");
    }

    // Map and copy shader handles to SBT buffer
    void* data;
    vkMapMemory(device->getDevice(), shaderBindingTableMemory, 0, shaderBindingTableSize, 0, &data);
    memcpy(data, shaderHandleStorage.data(), shaderBindingTableSize);
    vkUnmapMemory(device->getDevice(), shaderBindingTableMemory);
    */
    // only temporary!
}

void VolumeApp::createCommandBuffers() {
    commandBuffers.resize(swapChain->getImages().size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool->getCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(device->getDevice(), &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers!");
    }

    for (size_t i = 0; i < commandBuffers.size(); ++i) {
        recordCommandBuffer(commandBuffers[i], static_cast<uint32_t>(i));
    }
}

void VolumeApp::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    /*
    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }

    // Bind the ray tracing pipeline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rayTracingPipeline);

    // Bind the descriptor set
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        pipelineLayout,
        0,                      // First set
        1,                      // Descriptor set count
        &descriptorSet,         // Descriptor set
        0,                      // Dynamic offset count
        nullptr                 // Dynamic offsets
    );

    // Retrieve device address of SBT buffer
    VkBufferDeviceAddressInfo bufferDeviceAI{};
    bufferDeviceAI.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAI.buffer = shaderBindingTableBuffer;
    VkDeviceAddress sbtAddress = vkGetBufferDeviceAddress(device->getDevice(), &bufferDeviceAI);

    // Get properties for SBT
    const uint32_t handleSize = device->getRayTracingProperties().shaderGroupHandleSize;
    const uint32_t handleAlignment = device->getRayTracingProperties().shaderGroupHandleAlignment;
    const uint32_t groupBaseAlignment = device->getRayTracingProperties().shaderGroupBaseAlignment;

    // Calculate stride and size for each region
    VkDeviceSize groupSize = alignedSize(handleSize, handleAlignment);

    // Stride must be aligned to shaderGroupBaseAlignment
    VkDeviceSize stride = alignedSize(groupSize, groupBaseAlignment);

    // Define SBT regions
    VkStridedDeviceAddressRegionKHR raygenSbt{};
    raygenSbt.deviceAddress = sbtAddress;
    raygenSbt.stride = stride;
    raygenSbt.size = stride; // Only one raygen shader

    VkStridedDeviceAddressRegionKHR missSbt{};
    missSbt.deviceAddress = sbtAddress + stride; // Assuming miss shader follows raygen
    missSbt.stride = stride;
    missSbt.size = stride; // Only one miss shader

    VkStridedDeviceAddressRegionKHR hitSbt{}; // No hit shaders
    VkStridedDeviceAddressRegionKHR callableSbt{}; // No callable shaders

    // Dispatch rays
    vkCmdTraceRaysKHR(
        commandBuffer,
        &raygenSbt,
        &missSbt,
        &hitSbt,
        &callableSbt,
        swapChain->getExtent().width,
        swapChain->getExtent().height,
        1
    );

    // Transition storage image layout from GENERAL to TRANSFER_SRC_OPTIMAL
    VkImageMemoryBarrier storageImageBarrier{};
    storageImageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    storageImageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    storageImageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    storageImageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    storageImageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    storageImageBarrier.image = storageImage;
    storageImageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    storageImageBarrier.subresourceRange.baseMipLevel = 0;
    storageImageBarrier.subresourceRange.levelCount = 1;
    storageImageBarrier.subresourceRange.baseArrayLayer = 0;
    storageImageBarrier.subresourceRange.layerCount = 1;
    storageImageBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    storageImageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &storageImageBarrier
    );

    // Transition swap chain image layout from UNDEFINED to TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier swapChainImageBarrier{};
    swapChainImageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    swapChainImageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    swapChainImageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapChainImageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapChainImageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapChainImageBarrier.image = swapChain->getImages()[imageIndex];
    swapChainImageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    swapChainImageBarrier.subresourceRange.baseMipLevel = 0;
    swapChainImageBarrier.subresourceRange.levelCount = 1;
    swapChainImageBarrier.subresourceRange.baseArrayLayer = 0;
    swapChainImageBarrier.subresourceRange.layerCount = 1;
    swapChainImageBarrier.srcAccessMask = 0;
    swapChainImageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &swapChainImageBarrier
    );

    // Copy storage image to swap chain image
    VkImageCopy copyRegion{};
    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.baseArrayLayer = 0;
    copyRegion.srcSubresource.mipLevel = 0;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.srcOffset = { 0, 0, 0 };

    copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.dstSubresource.baseArrayLayer = 0;
    copyRegion.dstSubresource.mipLevel = 0;
    copyRegion.dstSubresource.layerCount = 1;
    copyRegion.dstOffset = { 0, 0, 0 };

    copyRegion.extent.width = swapChain->getExtent().width;
    copyRegion.extent.height = swapChain->getExtent().height;
    copyRegion.extent.depth = 1;

    vkCmdCopyImage(
        commandBuffer,
        storageImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapChain->getImages()[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &copyRegion
    );

    // Transition swap chain image layout to PRESENT_SRC_KHR
    VkImageMemoryBarrier swapChainImageBarrierPresent{};
    swapChainImageBarrierPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    swapChainImageBarrierPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapChainImageBarrierPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    swapChainImageBarrierPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapChainImageBarrierPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapChainImageBarrierPresent.image = swapChain->getImages()[imageIndex];
    swapChainImageBarrierPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    swapChainImageBarrierPresent.subresourceRange.baseMipLevel = 0;
    swapChainImageBarrierPresent.subresourceRange.levelCount = 1;
    swapChainImageBarrierPresent.subresourceRange.baseArrayLayer = 0;
    swapChainImageBarrierPresent.subresourceRange.layerCount = 1;
    swapChainImageBarrierPresent.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    swapChainImageBarrierPresent.dstAccessMask = 0;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &swapChainImageBarrierPresent
    );

    // Transition storage image layout back to GENERAL for next frame
    VkImageMemoryBarrier storageImageBarrierGeneral{};
    storageImageBarrierGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    storageImageBarrierGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    storageImageBarrierGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    storageImageBarrierGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    storageImageBarrierGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    storageImageBarrierGeneral.image = storageImage;
    storageImageBarrierGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    storageImageBarrierGeneral.subresourceRange.baseMipLevel = 0;
    storageImageBarrierGeneral.subresourceRange.levelCount = 1;
    storageImageBarrierGeneral.subresourceRange.baseArrayLayer = 0;
    storageImageBarrierGeneral.subresourceRange.layerCount = 1;
    storageImageBarrierGeneral.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    storageImageBarrierGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0,
        0, nullptr,
        0, nullptr,
        1, &storageImageBarrierGeneral
    );

    // End command buffer
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer!");
    }
    */
    // temp!
}

void VolumeApp::createSyncObjects() {
    syncObjects = std::make_unique<basalt::SyncObjects>(*device, MAX_FRAMES_IN_FLIGHT);
}

void VolumeApp::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }

    // Wait for device to finish operations before cleanup
    vkDeviceWaitIdle(device->getDevice());
}

void VolumeApp::drawFrame() {
    // Wait for the current frame's fence to be signaled
    syncObjects->waitForInFlightFence(currentFrame);

    // Acquire the next image from the swap chain
    uint32_t imageIndex;
    VkResult result = swapChain->acquireNextImage(*syncObjects, currentFrame, imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }

    // Reset the fence for the current frame to unsignaled state
    syncObjects->resetInFlightFence(currentFrame);

    // Prepare synchronization parameters
    const VkSemaphore waitSemaphores[] = { syncObjects->getImageAvailableSemaphore(currentFrame) };
    const VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR };
    const VkSemaphore signalSemaphores[] = { syncObjects->getRenderFinishedSemaphore(currentFrame) };

    // Submit the command buffer for execution
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(device->getGraphicsQueue(), 1, &submitInfo, syncObjects->getInFlightFence(currentFrame)) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer!");
    }

    // Present the rendered image to the swap chain
    result = swapChain->presentImage(*syncObjects, currentFrame, imageIndex);

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
    // Wait until the window is not minimized
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &width, &height);
    }

    // Wait for device to be idle before recreating swap chain
    vkDeviceWaitIdle(device->getDevice());

    // Cleanup existing swap chain resources
    swapChain->cleanup();

    // Recreate swap chain
    swapChain->recreateSwapChain(*device, *surface, window);

    // Recreate storage image
    vkDestroyImageView(device->getDevice(), storageImageView, nullptr);
    vkDestroyImage(device->getDevice(), storageImage, nullptr);
    vkFreeMemory(device->getDevice(), storageImageMemory, nullptr);
    createStorageImage();

    // Re-record command buffers with the new swap chain images
    vkFreeCommandBuffers(device->getDevice(), commandPool->getCommandPool(), static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
    createCommandBuffers();
}

void VolumeApp::cleanup() {
    // Wait for device to finish operations before cleanup
    vkDeviceWaitIdle(device->getDevice());

    // Cleanup resources
    vkDestroyDescriptorPool(device->getDevice(), descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device->getDevice(), descriptorSetLayout, nullptr);

    vkDestroyPipeline(device->getDevice(), rayTracingPipeline, nullptr);
    vkDestroyPipelineLayout(device->getDevice(), pipelineLayout, nullptr);

    vkDestroyBuffer(device->getDevice(), shaderBindingTableBuffer, nullptr);
    vkFreeMemory(device->getDevice(), shaderBindingTableMemory, nullptr);

    vkDestroyImageView(device->getDevice(), storageImageView, nullptr);
    vkDestroyImage(device->getDevice(), storageImage, nullptr);
    vkFreeMemory(device->getDevice(), storageImageMemory, nullptr);

    if (nanoVDBData) {
        free(nanoVDBData);
        nanoVDBData = nullptr;
    }

    // Resources managed by smart pointers will be automatically cleaned up

    // Destroy GLFW window and terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
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
