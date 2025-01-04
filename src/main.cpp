// #define MEASURE_PERFORMANCE
#define MEASURE_QUALITY

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <mutex>
#include <ctime>
#include <sstream>

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

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

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

constexpr uint32_t WIDTH = 1024;
constexpr uint32_t HEIGHT = 1024;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::string BEAM_LIGHT_GEN_PATH = "shaders/compiled_shaders/light_gen.comp.spv";
const std::string BEAM_COMPUTE_SHADER_PATH = "shaders/compiled_shaders/beam_compute_color.comp.spv";

const std::string RAY_LIGHT_GEN_PATH = "shaders/compiled_shaders/light_gen.comp.spv";
const std::string RAY_COMPUTE_SHADER_PATH = "shaders/compiled_shaders/ray_compute_color.comp.spv";

const std::string POINT_LIGHT_GEN_PATH = "shaders/compiled_shaders/light_gen.comp.spv";
const std::string POINT_COMPUTE_SHADER_PATH = "shaders/compiled_shaders/point_compute_color.comp.spv";

const std::string SPHERE_LIGHT_GEN_PATH = "shaders/compiled_shaders/light_gen.comp.spv";
const std::string SPHERE_COMPUTE_SHADER_PATH = "shaders/compiled_shaders/sphere_compute_color.comp.spv";

const std::string PATH_LIGHT_GEN_PATH = "shaders/compiled_shaders/path_light_gen.comp.spv";
const std::string PATH_COMPUTE_SHADER_PATH = "shaders/compiled_shaders/path_compute_color.comp.spv";

const std::string VERT_SHADER_PATH = "shaders/compiled_shaders/fullscreen.vert.spv";
const std::string FRAG_SHADER_PATH = "shaders/compiled_shaders/sample_image.frag.spv";

enum Algorithms
{
    BEAM, RAY, POINT, SPHERE, PATH
};

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
    alignas(4) float beamRadius;
    alignas(4) float lightRayStepSize;
    alignas(4) float radiusFalloff;
};

class VolumeApp {
public:
    VolumeApp() {
        initWindow();
        initVulkan();
        initImGui();
    }

    void run() {
        mainLoop();
        cleanup();
    }

private:
    // Window
    GLFWwindow* window;
    bool framebufferResized = false;

    // Vulkan components
    std::unique_ptr<basalt::Instance>     instance;
    std::unique_ptr<basalt::Surface>      surface;
    std::unique_ptr<basalt::Device>       device;
    std::unique_ptr<basalt::SwapChain>    swapChain;
    std::unique_ptr<basalt::RenderPass>   renderPass;
    std::unique_ptr<basalt::GraphicsPipeline> graphicsPipeline;
    std::unique_ptr<basalt::ComputePipeline> beamLightGenPipeline;
    std::unique_ptr<basalt::ComputePipeline> beamComputeColorPipeline;
    std::unique_ptr<basalt::ComputePipeline> rayLightGenPipeline;
    std::unique_ptr<basalt::ComputePipeline> rayComputeColorPipeline;
    std::unique_ptr<basalt::ComputePipeline> pointLightGenPipeline;
    std::unique_ptr<basalt::ComputePipeline> pointComputeColorPipeline;
    std::unique_ptr<basalt::ComputePipeline> sphereLightGenPipeline;
    std::unique_ptr<basalt::ComputePipeline> sphereComputeColorPipeline;
    std::unique_ptr<basalt::ComputePipeline> pathLightGenPipeline;
    std::unique_ptr<basalt::ComputePipeline> pathComputeColorPipeline;
    std::unique_ptr<basalt::CommandPool>   commandPool;
    std::unique_ptr<basalt::SyncObjects>   syncObjects;
    std::unique_ptr<basalt::DescriptorSetLayout> descriptorSetLayout;
    std::unique_ptr<basalt::DescriptorPool>      descriptorPool;
    VkDescriptorSet descriptorSet;

    // Storage image
    std::unique_ptr<basalt::Image> storageImage;
    VkFormat imageFormat;
    VkImageView storageImageView = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;

    // Buffers
    std::unique_ptr<basalt::Buffer> nanoVDBBuffer;
    std::unique_ptr<basalt::Buffer> rayLightsBuffer;
    std::unique_ptr<basalt::Buffer> lightCounterBuffer;
    std::unique_ptr<basalt::Buffer> uniformBuffer;
    UBO uboData{};

    // Pipelines
    VkPipelineLayout graphicsPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout computePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout lightGenPipelineLayout = VK_NULL_HANDLE;

    // Command buffers
    std::vector<std::unique_ptr<basalt::CommandBuffer>> commandBuffers;
    size_t currentFrame = 0;
    // A separate descriptor pool for ImGui
    VkDescriptorPool imguiDescriptorPool = VK_NULL_HANDLE;

    // Current algorithm
    Algorithms currentAlgorithm = RAY;

#if defined(MEASURE_PERFORMANCE)
    // ------------------------------------------------
    // PERFORMANCE MEASUREMENT
    // ------------------------------------------------
    // We'll store logs in something like "output/performance_<timestamp>.csv"
    std::string perfLogPath;
    std::ofstream perfLog;
    std::chrono::steady_clock::time_point frameStartTime;
    std::chrono::steady_clock::time_point frameEndTime;
    std::mutex perfMutex;

    // We'll save frames with exponential stepping:
    // but we still want a unique name each time (like a timestamp).
    int nextSaveFrame = 1;

    // Helper to get a timestamp string for the CSV file name
    static std::string getTimestampString()
    {
        auto now = std::time(nullptr);
        std::tm tmNow{};
#if defined(_WIN32)
        localtime_s(&tmNow, &now); // Windows
#else
        localtime_r(&now, &tmNow); // POSIX
#endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmNow);
        return std::string(buf);
    }

#elif defined(MEASURE_QUALITY)
    // QUALITY MEASUREMENT
    // We do not measure performance times at all.
    // We only save images if that file doesn't exist yet.
    int nextSaveFrame = 1; // same exponential stepping
#endif

    // For FPS in window title
    double lastTitleUpdateTime = 0.0;
    int framesSinceLastTitleUpdate = 0;

private:
    void initWindow();
    void initVulkan();
    void initImGui();

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

    void mainLoop();
    void drawFrame();
    void recreateSwapChain();
    void updateDescriptorSet() const;
    void cleanupSwapChain();
    void cleanup();
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

    // Returns correct pipeline based on currentAlgorithm
    VkPipeline getCurrentLightGenVkPipeline() const
    {
        switch (currentAlgorithm)
        {
        case BEAM:
            return beamLightGenPipeline->getPipeline();
        case RAY:
            return rayLightGenPipeline->getPipeline();
        case POINT:
            return pointLightGenPipeline->getPipeline();
        case SPHERE:
            return sphereLightGenPipeline->getPipeline();
        default:
            return pathLightGenPipeline->getPipeline();
        }
    }
    VkPipeline getCurrentComputeColorVkPipeline() const
    {
        switch (currentAlgorithm)
        {
        case BEAM:
            return beamComputeColorPipeline->getPipeline();
        case RAY:
            return rayComputeColorPipeline->getPipeline();
        case POINT:
            return pointComputeColorPipeline->getPipeline();
        case SPHERE:
            return sphereComputeColorPipeline->getPipeline();
        default:
            return pathComputeColorPipeline->getPipeline();
        }
    }

    // --- Helper to save image from GPU
    void saveImage(const std::string& fileName);

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

#if defined(MEASURE_PERFORMANCE)
    // Build a performance CSV path in the output folder, e.g. "output/performance_YYYYMMDD_HHMMSS.csv"
    {
        auto outDir = std::filesystem::path("output");
        std::filesystem::create_directories(outDir); // Ensure "output" exists
        perfLogPath = (outDir / ("performance_" + getTimestampString() + ".csv")).string();
        perfLog.open(perfLogPath, std::ios::out);
        if (!perfLog.is_open()) {
            throw std::runtime_error("Failed to open performance log file: " + perfLogPath);
        }
        // Possibly write a header line
        perfLog << "Algorithm,FrameTime_ms\n";
    }
#endif
}

void VolumeApp::initImGui() {
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 },
        { VK_DESCRIPTOR_TYPE_SAMPLER,                100 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          100 },
    };
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes));
    pool_info.pPoolSizes = pool_sizes;
    pool_info.maxSets = 100;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    if (vkCreateDescriptorPool(device->getDevice(), &pool_info, nullptr, &imguiDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ImGui descriptor pool!");
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.Instance = instance->getInstance();
    initInfo.PhysicalDevice = device->getPhysicalDevice();
    initInfo.Device = device->getDevice();
    initInfo.QueueFamily = device->getGraphicsQueueFamilyIndex();
    initInfo.Queue = device->getGraphicsQueue();
    initInfo.DescriptorPool = imguiDescriptorPool;
    initInfo.RenderPass = renderPass->getRenderPass();
    initInfo.PipelineCache = VK_NULL_HANDLE;
    initInfo.Subpass = 0;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = static_cast<uint32_t>(swapChain->getFramebuffers().size());
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.Allocator = nullptr;
    initInfo.CheckVkResultFn = nullptr;
    ImGui_ImplVulkan_Init(&initInfo);

    ImGui_ImplVulkan_CreateFontsTexture();
}

void VolumeApp::createStorageImage() {
    imageFormat = basalt::utils::findSupportedFormat(*device,
        { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM },
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);

    storageImage = std::make_unique<basalt::Image>(
        *device,
        swapChain->getExtent().width,
        swapChain->getExtent().height,
        imageFormat,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
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

    basalt::utils::transitionImageLayout(
        *device,
        *commandPool,
        device->getGraphicsQueue(),
        storageImage->getImage(),
        imageFormat,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL);
}

void VolumeApp::createLightBuffers() {
    VkDeviceSize maxLights = 100000;
    VkDeviceSize pointLightsBufferSize = sizeof(RayLight) * maxLights;
    rayLightsBuffer = std::make_unique<basalt::Buffer>(
        *device, pointLightsBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkDeviceSize counterBufferSize = sizeof(LightCountBuffer);
    lightCounterBuffer = std::make_unique<basalt::Buffer>(
        *device, counterBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
}

void VolumeApp::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding storageImageLayoutBinding{};
    storageImageLayoutBinding.binding = 0;
    storageImageLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    storageImageLayoutBinding.descriptorCount = 1;
    storageImageLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding storageBufferLayoutBinding{};
    storageBufferLayoutBinding.binding = 2;
    storageBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    storageBufferLayoutBinding.descriptorCount = 1;
    storageBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding pointLightsBinding{};
    pointLightsBinding.binding = 3;
    pointLightsBinding.descriptorCount = 1;
    pointLightsBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pointLightsBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding lightCounterBinding{};
    lightCounterBinding.binding = 4;
    lightCounterBinding.descriptorCount = 1;
    lightCounterBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    lightCounterBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding uniformBufferLayoutBinding{};
    uniformBufferLayoutBinding.binding = 5;
    uniformBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBufferLayoutBinding.descriptorCount = 1;
    uniformBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

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
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
    };

    descriptorPool = std::make_unique<basalt::DescriptorPool>(*device, 1, poolSizes);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool->getPool();
    allocInfo.descriptorSetCount = 1;
    VkDescriptorSetLayout layout = descriptorSetLayout->getLayout();
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

    beamLightGenPipeline = std::make_unique<basalt::ComputePipeline>(*device, BEAM_LIGHT_GEN_PATH, lightGenPipelineLayout);
    rayLightGenPipeline = std::make_unique<basalt::ComputePipeline>(*device, RAY_LIGHT_GEN_PATH, lightGenPipelineLayout);
    pointLightGenPipeline = std::make_unique<basalt::ComputePipeline>(*device, POINT_LIGHT_GEN_PATH, lightGenPipelineLayout);
    sphereLightGenPipeline = std::make_unique<basalt::ComputePipeline>(*device, SPHERE_LIGHT_GEN_PATH, lightGenPipelineLayout);
    pathLightGenPipeline = std::make_unique<basalt::ComputePipeline>(*device, PATH_LIGHT_GEN_PATH, lightGenPipelineLayout);
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

    beamComputeColorPipeline = std::make_unique<basalt::ComputePipeline>(*device, BEAM_COMPUTE_SHADER_PATH, computePipelineLayout);
    rayComputeColorPipeline = std::make_unique<basalt::ComputePipeline>(*device, RAY_COMPUTE_SHADER_PATH, computePipelineLayout);
    pointComputeColorPipeline = std::make_unique<basalt::ComputePipeline>(*device, POINT_COMPUTE_SHADER_PATH, computePipelineLayout);
    sphereComputeColorPipeline = std::make_unique<basalt::ComputePipeline>(*device, SPHERE_COMPUTE_SHADER_PATH, computePipelineLayout);
    pathComputeColorPipeline = std::make_unique<basalt::ComputePipeline>(*device, PATH_COMPUTE_SHADER_PATH, computePipelineLayout);
}

void VolumeApp::createGraphicsPipeline() {
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

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
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        commandBuffers[i] = std::make_unique<basalt::CommandBuffer>(*device, *commandPool);
    }
}

void VolumeApp::createSyncObjects() {
    syncObjects = std::make_unique<basalt::SyncObjects>(*device, MAX_FRAMES_IN_FLIGHT);
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

    if (vkCreateSampler(device->getDevice(), &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sampler!");
    }
}

void VolumeApp::createUniformBuffer() {
    VkDeviceSize bufferSize = sizeof(UBO);
    uniformBuffer = std::make_unique<basalt::Buffer>(*device, bufferSize,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    uboData.frameCount = 0;
    uboData.framebufferDim = glm::uvec2(swapChain->getExtent().width, swapChain->getExtent().height);
    uboData.cameraPos = glm::vec3(0.0, 20.0, -75.0);
    uboData.fov = 45.0f;
    uboData.photonInitialIntensity = 100.0f;
    uboData.scatteringProbability = 0.05f;
    uboData.absorptionCoefficient = 0.05f;
    uboData.maxLights = 1000;
    uboData.rayMaxDistance = 2500.0f;
    uboData.rayMarchingStepSize = 1.0f;
    uboData.lightSourceWorldPos = glm::vec3(-20.0, 15.0, -15.0);
    uboData.beamRadius = 0.1f;
    uboData.lightRayStepSize = 0.3f;
    uboData.radiusFalloff = 0.5f;

    uniformBuffer->updateBuffer(*commandPool, &uboData, sizeof(UBO));
}

void VolumeApp::createNanoVDBBuffer() {
    openvdb::initialize();
    const std::string filename = "../../../../resources/bunny_cloud.vdb";

    openvdb::io::File file(filename);
    try {
        file.open();
    }
    catch (const openvdb::IoError& e) {
        std::cerr << "Error opening file " << filename << ": " << e.what() << '\n';
        throw std::runtime_error("Failed to open VDB file.");
    }

    openvdb::GridBase::Ptr baseGrid_openvdb;
    openvdb::FloatGrid::Ptr grid_openvdb;

    for (auto nameIter = file.beginName(); nameIter != file.endName(); ++nameIter) {
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

    auto handle = nanovdb::tools::createNanoGrid(*grid_openvdb);
    const void* gridData = handle.data();
    size_t gridSize = handle.size();

    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(gridSize);

    nanoVDBBuffer = std::make_unique<basalt::Buffer>(
        *device,
        bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    basalt::Buffer stagingBuffer(
        *device,
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    stagingBuffer.updateBuffer(*commandPool, gridData, bufferSize);

    basalt::utils::copyBuffer(*device, *commandPool, device->getGraphicsQueue(),
        stagingBuffer.getBuffer(), nanoVDBBuffer->getBuffer(), bufferSize);

    openvdb::uninitialize();
}

void VolumeApp::updateDescriptorSet() const {
    std::array<VkWriteDescriptorSet, 6> descriptorWrites{};

    // (0) Storage image
    VkDescriptorImageInfo storageImageInfo{};
    storageImageInfo.imageView = storageImageView;
    storageImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pImageInfo = &storageImageInfo;

    // (1) Combined image sampler
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

    // (2) NanoVDB buffer
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

    // (3) Ray lights buffer
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

    // (4) Light count buffer
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

    // (5) Uniform buffer
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

    vkUpdateDescriptorSets(device->getDevice(),
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0, nullptr);
}

void VolumeApp::saveImage(const std::string& fileName)
{
    VkDeviceSize imageSize = swapChain->getExtent().width * swapChain->getExtent().height * 4;

    basalt::Buffer stagingBuffer(
        *device,
        imageSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    auto& cmdBuffer = *commandBuffers[currentFrame];
    cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = storageImage->getImage();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        *cmdBuffer.get(),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { swapChain->getExtent().width, swapChain->getExtent().height, 1 };

    vkCmdCopyImageToBuffer(
        *cmdBuffer.get(),
        storageImage->getImage(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        stagingBuffer.getBuffer(),
        1,
        &region
    );

    // Transition back
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        *cmdBuffer.get(),
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    cmdBuffer.end();

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkCommandBuffer cb = *cmdBuffer.get();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cb;

    vkResetFences(device->getDevice(), 1, &syncObjects->getInFlightFence(currentFrame));
    if (vkQueueSubmit(device->getGraphicsQueue(), 1, &submitInfo, syncObjects->getInFlightFence(currentFrame)) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit copy for screenshot!");
    }

    vkWaitForFences(device->getDevice(), 1, &syncObjects->getInFlightFence(currentFrame), VK_TRUE, UINT64_MAX);

    void* data;
    vkMapMemory(device->getDevice(), stagingBuffer.getBufferMemory(), 0, imageSize, 0, &data);

    // Make directories if needed
    auto parentPath = std::filesystem::path(fileName).parent_path();
    if (!parentPath.empty()) {
        std::filesystem::create_directories(parentPath);
    }

    // Actually write out the PNG
    stbi_write_png(
        fileName.c_str(),
        static_cast<int>(swapChain->getExtent().width),
        static_cast<int>(swapChain->getExtent().height),
        4,
        data,
        static_cast<int>(swapChain->getExtent().width) * 4
    );

    vkUnmapMemory(device->getDevice(), stagingBuffer.getBufferMemory());
}

void VolumeApp::mainLoop() {
    double startTime = glfwGetTime();
    lastTitleUpdateTime = startTime;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();

        // Update FPS once per second
        double currentTime = glfwGetTime();
        framesSinceLastTitleUpdate++;
        if (currentTime - lastTitleUpdateTime >= 1.0) {
            double fps = framesSinceLastTitleUpdate / (currentTime - lastTitleUpdateTime);
            char title[256];
            std::snprintf(title, sizeof(title), "Volume Renderer - %d [FPS: %.2f]", uboData.frameCount, fps);
            glfwSetWindowTitle(window, title);
            framesSinceLastTitleUpdate = 0;
            lastTitleUpdateTime = currentTime;
        }
    }
    vkDeviceWaitIdle(device->getDevice());
}

void VolumeApp::drawFrame() {
#if defined(MEASURE_PERFORMANCE)
    frameStartTime = std::chrono::steady_clock::now();
#endif

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

    // ImGui
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Settings");
    {
        static const char* algorithmNames[] = { "Beam", "Ray", "Point", "Sphere", "Path" };
        int currentItem = static_cast<int>(currentAlgorithm);
        if (ImGui::Combo("Algorithm", &currentItem, algorithmNames, IM_ARRAYSIZE(algorithmNames))) {
            currentAlgorithm = static_cast<Algorithms>(currentItem);
            uboData.frameCount = 0;
#if defined(MEASURE_PERFORMANCE) || defined(MEASURE_QUALITY)
            nextSaveFrame = 1;
#endif
        }
    }
    ImGui::SliderFloat3("Camera Pos", &uboData.cameraPos.x, -200.0f, 200.0f);
    ImGui::SliderFloat("Photon Intensity", &uboData.photonInitialIntensity, 0.0f, 500.0f);
    ImGui::SliderFloat("Scattering Probability", &uboData.scatteringProbability, 0.0f, 1.0f);
    ImGui::SliderFloat("Absorption Coeff", &uboData.absorptionCoefficient, 0.0f, 1.0f);
    int maxLightsTmp = static_cast<int>(uboData.maxLights);
    if (ImGui::DragInt("Max Lights", &maxLightsTmp, 1.0f, 0, 1000000)) {
        uboData.maxLights = static_cast<uint32_t>(maxLightsTmp);
    }
    ImGui::SliderFloat("Ray Max Dist", &uboData.rayMaxDistance, 0.0f, 20000.0f);
    ImGui::SliderFloat("Ray Step Size", &uboData.rayMarchingStepSize, 0.0f, 10.0f);
    ImGui::SliderFloat3("Light Source Pos", &uboData.lightSourceWorldPos.x, -100.0f, 100.0f);
    ImGui::SliderFloat("Beam Radius", &uboData.beamRadius, 0.0f, 10.0f);
    ImGui::SliderFloat("Light Step Size", &uboData.lightRayStepSize, 0.0f, 10.0f);

    if (ImGui::Button("Refresh")) {
        uboData.frameCount = 0;
#if defined(MEASURE_PERFORMANCE) || defined(MEASURE_QUALITY)
        nextSaveFrame = 1;
#endif
    }
    ImGui::End();
    ImGui::Render();

    // Update UBO
    uboData.frameCount++;
    uniformBuffer->updateBuffer(*commandPool, &uboData, sizeof(UBO));

    // Record command buffer
    auto& cmdBuffer = *commandBuffers[currentFrame];
    cmdBuffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    // Reset the light counter
    vkCmdFillBuffer(
        *cmdBuffer.get(),
        lightCounterBuffer->getBuffer(),
        0,
        sizeof(LightCountBuffer),
        0
    );

    // Clear image if first frame
    if (uboData.frameCount == 1) {
        VkClearColorValue clearColorValue{};
        clearColorValue.float32[0] = 0.0f;
        clearColorValue.float32[1] = 0.0f;
        clearColorValue.float32[2] = 0.0f;
        clearColorValue.float32[3] = 1.0f;

        VkImageSubresourceRange subresourceRange{};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = storageImage->getImage();
        barrier.subresourceRange = subresourceRange;

        vkCmdPipelineBarrier(
            *cmdBuffer.get(),
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        vkCmdClearColorImage(*cmdBuffer.get(),
            storageImage->getImage(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            &clearColorValue,
            1, &subresourceRange);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(
            *cmdBuffer.get(),
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
    }

    // 1) Light generation
    vkCmdBindPipeline(*cmdBuffer.get(), VK_PIPELINE_BIND_POINT_COMPUTE, getCurrentLightGenVkPipeline());
    vkCmdBindDescriptorSets(*cmdBuffer.get(), VK_PIPELINE_BIND_POINT_COMPUTE, lightGenPipelineLayout,
        0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(*cmdBuffer.get(), 1, 1, 1);

    VkMemoryBarrier memoryBarrier1{};
    memoryBarrier1.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier1.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        *cmdBuffer.get(),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &memoryBarrier1,
        0, nullptr,
        0, nullptr
    );

    // 2) Main compute
    vkCmdBindPipeline(*cmdBuffer.get(), VK_PIPELINE_BIND_POINT_COMPUTE, getCurrentComputeColorVkPipeline());
    vkCmdBindDescriptorSets(*cmdBuffer.get(), VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout,
        0, 1, &descriptorSet, 0, nullptr);
    uint32_t groupCountX = (swapChain->getExtent().width + 15) / 16;
    uint32_t groupCountY = (swapChain->getExtent().height + 15) / 16;
    vkCmdDispatch(*cmdBuffer.get(), groupCountX, groupCountY, 1);

    // Barrier
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
        0, nullptr,
        0, nullptr,
        1, &imageBarrier
    );

    // 3) Render pass
    VkClearValue clearValue{};
    clearValue.color = { { 0.f, 0.f, 0.f, 1.f } };
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass->getRenderPass();
    renderPassInfo.framebuffer = swapChain->getFramebuffers()[imageIndex];
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = swapChain->getExtent();
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearValue;

    vkCmdBeginRenderPass(*cmdBuffer.get(), &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // 4) Fullscreen
    vkCmdBindPipeline(*cmdBuffer.get(), VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline->getPipeline());
    vkCmdBindDescriptorSets(*cmdBuffer.get(), VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout,
        0, 1, &descriptorSet, 0, nullptr);
    vkCmdDraw(*cmdBuffer.get(), 3, 1, 0, 0);

    // 5) ImGui
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cmdBuffer.get());
    vkCmdEndRenderPass(*cmdBuffer.get());

    cmdBuffer.end();

    // Submit
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = { syncObjects->getImageAvailableSemaphore(currentFrame) };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    VkCommandBuffer submitCmdBuffer = *cmdBuffer.get();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &submitCmdBuffer;

    VkSemaphore signalSemaphores[] = { syncObjects->getRenderFinishedSemaphore(currentFrame) };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(device->getGraphicsQueue(), 1, &submitInfo, syncObjects->getInFlightFence(currentFrame)) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer!");
    }

    // Present
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
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

#if defined(MEASURE_PERFORMANCE)
    frameEndTime = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(frameEndTime - frameStartTime).count();

    if (currentAlgorithm == RAY || currentAlgorithm == POINT || currentAlgorithm == PATH) {
        std::lock_guard<std::mutex> lock(perfMutex);
        if (perfLog.is_open()) {
            perfLog << (currentAlgorithm == RAY ? "Ray" :
                currentAlgorithm == POINT ? "Point" :
                "Path")
                << "," << ms << "\n";
        }
    }

    // Save screenshot on exponential frames
    if (currentAlgorithm == RAY || currentAlgorithm == POINT || currentAlgorithm == PATH) {
        if (static_cast<int>(uboData.frameCount) == nextSaveFrame) {
            // We'll incorporate a timestamp into the file name
            // e.g. output/ray/frame_16_20250104_123456.png
            auto nowStr = getTimestampString();
            std::string folderName = (currentAlgorithm == RAY ? "output/ray" :
                (currentAlgorithm == POINT ? "output/point" :
                    "output/path"));
            std::stringstream ss;
            ss << folderName << "/frame_" << uboData.frameCount << "_" << nowStr << ".png";

            saveImage(ss.str());
            nextSaveFrame *= 2;
        }
    }

#elif defined(MEASURE_QUALITY)
    if (currentAlgorithm == RAY || currentAlgorithm == POINT || currentAlgorithm == PATH) {
        if (static_cast<int>(uboData.frameCount) == nextSaveFrame) {
            // We only save if the file doesn't exist
            std::string folderName = (currentAlgorithm == RAY ? "output/ray" :
                (currentAlgorithm == POINT ? "output/point" :
                    "output/path"));
            std::stringstream ss;
            ss << folderName << "/frame_" << uboData.frameCount << ".png";
            auto outPath = ss.str();

            if (!std::filesystem::exists(outPath)) {
                saveImage(outPath);
            }
            nextSaveFrame *= 2;
        }
    }
#endif

    currentFrame = (currentFrame + 1) % syncObjects->getMaxFramesInFlight();
}

void VolumeApp::recreateSwapChain() {
    uboData.frameCount = 0;

#if defined(MEASURE_PERFORMANCE) || defined(MEASURE_QUALITY)
    nextSaveFrame = 1;
#endif

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
    createLightGenPipeline();
    createComputePipeline();
    createGraphicsPipeline();
    swapChain->createFramebuffers(*renderPass);
    createCommandBuffers();

    ImGui_ImplVulkan_SetMinImageCount(2);
    ImGui_ImplVulkan_Shutdown();

    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.Instance = instance->getInstance();
    initInfo.PhysicalDevice = device->getPhysicalDevice();
    initInfo.Device = device->getDevice();
    initInfo.QueueFamily = device->getGraphicsQueueFamilyIndex();
    initInfo.Queue = device->getGraphicsQueue();
    initInfo.DescriptorPool = imguiDescriptorPool;
    initInfo.RenderPass = renderPass->getRenderPass();
    initInfo.PipelineCache = VK_NULL_HANDLE;
    initInfo.Subpass = 0;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = static_cast<uint32_t>(swapChain->getFramebuffers().size());
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.Allocator = nullptr;
    initInfo.CheckVkResultFn = nullptr;
    ImGui_ImplVulkan_Init(&initInfo);
    ImGui_ImplVulkan_CreateFontsTexture();
}

void VolumeApp::cleanupSwapChain() {
    vkDeviceWaitIdle(device->getDevice());

    commandBuffers.clear();

    if (storageImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device->getDevice(), storageImageView, nullptr);
        storageImageView = VK_NULL_HANDLE;
    }
    storageImage.reset();

    beamLightGenPipeline.reset();
    beamComputeColorPipeline.reset();
    rayLightGenPipeline.reset();
    rayComputeColorPipeline.reset();
    pointLightGenPipeline.reset();
    pointComputeColorPipeline.reset();
    sphereLightGenPipeline.reset();
    sphereComputeColorPipeline.reset();
    pathLightGenPipeline.reset();
    pathComputeColorPipeline.reset();
    graphicsPipeline.reset();

    if (lightGenPipelineLayout) {
        vkDestroyPipelineLayout(device->getDevice(), lightGenPipelineLayout, nullptr);
        lightGenPipelineLayout = VK_NULL_HANDLE;
    }
    if (graphicsPipelineLayout) {
        vkDestroyPipelineLayout(device->getDevice(), graphicsPipelineLayout, nullptr);
        graphicsPipelineLayout = VK_NULL_HANDLE;
    }
    if (computePipelineLayout) {
        vkDestroyPipelineLayout(device->getDevice(), computePipelineLayout, nullptr);
        computePipelineLayout = VK_NULL_HANDLE;
    }

    swapChain->cleanup();
    renderPass.reset();
    swapChain.reset();
}

void VolumeApp::cleanup() {
    vkDeviceWaitIdle(device->getDevice());

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (imguiDescriptorPool) {
        vkDestroyDescriptorPool(device->getDevice(), imguiDescriptorPool, nullptr);
        imguiDescriptorPool = VK_NULL_HANDLE;
    }

    if (storageImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device->getDevice(), storageImageView, nullptr);
        storageImageView = VK_NULL_HANDLE;
    }

    if (sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device->getDevice(), sampler, nullptr);
        sampler = VK_NULL_HANDLE;
    }

    storageImage.reset();
    nanoVDBBuffer.reset();
    rayLightsBuffer.reset();
    lightCounterBuffer.reset();
    uniformBuffer.reset();

    cleanupSwapChain();

    descriptorPool.reset();
    descriptorSetLayout.reset();

    commandBuffers.clear();
    commandPool.reset();
    syncObjects.reset();

    device.reset();
    surface.reset();
    instance.reset();

    glfwDestroyWindow(window);
    glfwTerminate();

#if defined(MEASURE_PERFORMANCE)
    if (perfLog.is_open()) {
        perfLog.close();
    }
#endif
}

void VolumeApp::framebufferResizeCallback(GLFWwindow* wnd, int width, int height) {
    auto app = reinterpret_cast<VolumeApp*>(glfwGetWindowUserPointer(wnd));
    app->framebufferResized = true;
}

int main() {
    try {
        VolumeApp app;
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::cin.get(); // pause for debugging
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
