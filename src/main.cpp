#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <cstdlib>

#include <vulkan/vulkan_core.h>
#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>

#include "buffer.h"
#include "command_buffer.h"
#include "command_pool.h"
#include "device.h"
#include "instance.h"
#include "pipeline.h"
#include "renderpass.h"
#include "simple_vertex_2D.h"
#include "surface.h"
#include "swapchain.h"
#include "sync_objects.h"

// Constants for window dimensions
constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

// Maximum number of frames that can be processed concurrently
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

// Paths to compiled shader modules
const std::string VERT_SHADER_PATH = "shaders/compiled_shaders/triangle.vert.spv";
const std::string FRAG_SHADER_PATH = "shaders/compiled_shaders/triangle.frag.spv";

// Vertex data for a simple triangle
std::vector<basalt::SimpleVertex2D> vertices = {
    {{ 0.0f, -0.5f}, { 1.0f, 0.0f, 0.0f }},  // Bottom vertex (Red)
    {{ 0.5f,  0.5f}, { 0.0f, 1.0f, 0.0f }},  // Right vertex (Green)
    {{-0.5f,  0.5f}, { 0.0f, 0.0f, 1.0f }}   // Left vertex (Blue)
};

class BasaltApp {
public:
    BasaltApp() {
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
    std::unique_ptr<basalt::RenderPass> renderPass;
    std::unique_ptr<basalt::Pipeline> pipeline;
    std::unique_ptr<basalt::CommandPool> commandPool;
    std::unique_ptr<basalt::Buffer> vertexBuffer;
    std::unique_ptr<basalt::SyncObjects> syncObjects;

    // Command buffers
    std::vector<std::unique_ptr<basalt::CommandBuffer>> commandBuffers;

    // Frame management
    size_t currentFrame = 0;
    bool framebufferResized = false;

    // Initialization methods
    void initWindow();
    void initVulkan();
    void createVertexBuffer();
    void createCommandBuffers();
    void createSyncObjects();

    // Rendering loop
    void mainLoop();

    // Frame rendering
    void drawFrame();

    // Swap chain recreation handled within SwapChain class
    void recreateSwapChain();
    void cleanup() const;

    // Callback for framebuffer resize
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        const auto app = static_cast<BasaltApp*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
};

void BasaltApp::initWindow() {
    // Initialize GLFW
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW!");
    }

    // No default API (Vulkan)
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    // Create GLFW window
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Triangle", nullptr, nullptr);
    if (!window) {
        throw std::runtime_error("Failed to create GLFW window!");
    }

    // Set the user pointer to this class instance for callback access
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void BasaltApp::initVulkan() {
    // Create Vulkan instance
    instance = std::make_unique<basalt::Instance>();

    // Create surface associated with the window
    surface = std::make_unique<basalt::Surface>(*instance, window);

    // Create logical device and retrieve queues
    device = std::make_unique<basalt::Device>(*instance, *surface);

    // Create swap chain
    swapChain = std::make_unique<basalt::SwapChain>(*device, *surface, window);

    // Create render pass
    renderPass = std::make_unique<basalt::RenderPass>(*device, swapChain->getImageFormat());

    // Create graphics pipeline
    VkVertexInputBindingDescription bindingDescription = basalt::SimpleVertex2D::getBindingDescription();
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions = basalt::SimpleVertex2D::getAttributeDescriptions();

    pipeline = std::make_unique<basalt::Pipeline>(*device, *renderPass, *swapChain, VERT_SHADER_PATH, FRAG_SHADER_PATH,
        bindingDescription, attributeDescriptions);

    // Create command pool
    commandPool = std::make_unique<basalt::CommandPool>(*device, device->getGraphicsQueueFamilyIndex());

    // Create vertex buffer and upload vertex data
    createVertexBuffer();

    // Create framebuffers for the swap chain images
    swapChain->createFramebuffers(*renderPass);

    // Allocate and record command buffers
    createCommandBuffers();

    // Create synchronization objects
    createSyncObjects();
}

void BasaltApp::createVertexBuffer()
{
    const VkDeviceSize vertexBufferSize = sizeof(vertices[0]) * vertices.size();
    vertexBuffer = std::make_unique<basalt::Buffer>(*device, vertexBufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vertexBuffer->updateBuffer(*commandPool, reinterpret_cast<void*>(vertices.data()), vertexBufferSize);
}

void BasaltApp::createCommandBuffers() {
    commandBuffers.resize(swapChain->getFramebuffers().size());

    for (size_t i = 0; i < commandBuffers.size(); ++i) {
        commandBuffers[i] = std::make_unique<basalt::CommandBuffer>(*device, *commandPool);

        commandBuffers[i]->begin(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

        constexpr VkClearValue clearColor = { {0.0f, 0.0f, 0.0f, 1.0f} };
        commandBuffers[i]->beginRenderPass(renderPass->getRenderPass(), swapChain->getFramebuffers()[i], swapChain->getExtent(), clearColor);

        commandBuffers[i]->bindPipeline(pipeline->getPipeline());
        commandBuffers[i]->bindVertexBuffer(vertexBuffer->getBuffer());

        commandBuffers[i]->draw(static_cast<uint32_t>(vertices.size()));

        commandBuffers[i]->endRenderPass();
        commandBuffers[i]->end();
    }
}

void BasaltApp::createSyncObjects() {
    syncObjects = std::make_unique<basalt::SyncObjects>(*device, MAX_FRAMES_IN_FLIGHT);
}

void BasaltApp::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }

    // Wait for device to finish operations before cleanup
    vkDeviceWaitIdle(device->getDevice());
}

void BasaltApp::drawFrame() {
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
    constexpr VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    const VkSemaphore signalSemaphores[] = { syncObjects->getRenderFinishedSemaphore(currentFrame) };

    // Submit the command buffer for execution using Device method
    if (device->submitCommandBuffers(
        commandBuffers[imageIndex]->get(), 1,
        waitSemaphores, 1,
        waitStages,
        signalSemaphores, 1,
        syncObjects->getInFlightFence(currentFrame)) != VK_SUCCESS) {
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

void BasaltApp::recreateSwapChain() {
    // Wait until the window is not minimized
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &width, &height);
    }

    // Wait for device to be idle before recreating swap chain
    vkDeviceWaitIdle(device->getDevice());

    // Recreate swap chain
    swapChain->recreateSwapChain(*device, *surface, window);

    // Recreate render pass (if it depends on swap chain format)
    renderPass = std::make_unique<basalt::RenderPass>(*device, swapChain->getImageFormat());

    // Recreate graphics pipeline with the new render pass and swap chain
    VkVertexInputBindingDescription bindingDescription = basalt::SimpleVertex2D::getBindingDescription();
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions = basalt::SimpleVertex2D::getAttributeDescriptions();

    pipeline = std::make_unique<basalt::Pipeline>(*device, *renderPass, *swapChain, VERT_SHADER_PATH, FRAG_SHADER_PATH,
        bindingDescription, attributeDescriptions);

    // Recreate framebuffers
    swapChain->createFramebuffers(*renderPass);

    // Re-record command buffers with the new framebuffers and pipeline
    commandBuffers.clear();
    commandBuffers.resize(swapChain->getFramebuffers().size());
    createCommandBuffers();
}

void BasaltApp::cleanup() const {
    // Wait for device to finish operations before cleanup
    vkDeviceWaitIdle(device->getDevice());

    // Resources are automatically cleaned up by smart pointers

    // Destroy GLFW window and terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
}

int main() {
    try {
        BasaltApp app;
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
