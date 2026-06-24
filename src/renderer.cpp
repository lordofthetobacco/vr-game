#include "include/renderer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <meshoptimizer.h>

#include "include/io.hpp"
#include "include/player.hpp"
#include "include/xr.hpp"
#include "third_party/stb_image.h"

namespace
{

    const char *kValidationLayer = "VK_LAYER_KHRONOS_validation";

    void vkCheck(VkResult result, const char *what)
    {
        if (result != VK_SUCCESS)
        {
            std::string msg = std::string("Vulkan error (") + std::to_string(result) +
                              ") in " + what;
            throw std::runtime_error(msg);
        }
    }

    bool validationLayerAvailable()
    {
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> layers(count);
        vkEnumerateInstanceLayerProperties(&count, layers.data());
        for (const auto &l : layers)
        {
            if (std::strcmp(l.layerName, kValidationLayer) == 0)
            {
                return true;
            }
        }
        return false;
    }

    // GPU vertex layout after 16-bit/8-bit quantization. 20 bytes vs 48 for the
    // in-memory float Vertex. Decoded for free by Vulkan vertex fetch; position is
    // dequantized in the vertex shader using the UBO bbox.
    struct PackedVertex
    {
        uint16_t pos[4];   // R16G16B16A16_UNORM (bbox-normalized; w unused)
        int8_t normal[4];  // R8G8B8A8_SNORM (w unused)
        int8_t tangent[4]; // R8G8B8A8_SNORM (xyz tangent, w = handedness sign)
        uint16_t uv[2];    // R16G16_SFLOAT
    };

    struct PushConstants
    {
        glm::mat4 model;
    };

    VkSampleCountFlagBits getMaxSampleCount(VkPhysicalDevice dev,
                                            VkFormat colorFormat)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);

        VkSampleCountFlags counts = props.limits.framebufferColorSampleCounts &
                                    props.limits.framebufferDepthSampleCounts;

        // Intersect with what the specific color format actually supports as a
        // multisampled attachment — device limits alone don't account for this.
        VkImageFormatProperties fmtProps{};
        if (vkGetPhysicalDeviceImageFormatProperties(
                dev, colorFormat, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                0, &fmtProps) == VK_SUCCESS)
        {
            counts &= fmtProps.sampleCounts;
        }

        for (auto c : {VK_SAMPLE_COUNT_8_BIT, VK_SAMPLE_COUNT_4_BIT,
                       VK_SAMPLE_COUNT_2_BIT})
            if (counts & c)
                return c;
        return VK_SAMPLE_COUNT_1_BIT;
    }

} // namespace

Renderer::Renderer(SDL_Window *window, XrSystem *xrSystem) : window(window)
{
    xr = (xrSystem && xrSystem->available()) ? xrSystem : nullptr;

    createInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();

    // With OpenXR, bind the session to the device and set up XR swapchains/input
    // before building the rest so XR resources can be created up front.
    if (xr)
    {
        xr->createSession(instance, physicalDevice, device, graphicsFamily, 0);
        xr->createReferenceSpace();
        // Preferred XR color formats (SRGB). The render pass targets whatever
        // the runtime picks from this list.
        xrColorFormat = xr->createSwapchains(
            {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB});
        xr->createInput();
    }

    createSwapchain();
    msaaSamples = getMaxSampleCount(physicalDevice, swapchainFormat);
    createImageViews();
    createMsaaColorResources();
    createDepthResources();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createUniformBuffers();
    createTextureSampler();
    createDescriptorPool();
    createCommandBuffers();
    createSyncObjects();
    initImGui();

    if (xr)
    {
        createXrRenderPass();
        xrPipeline = buildPipeline(xrRenderPass, msaaSamples);
        createXrRenderTargets();
        createXrFrameResources();
    }
}

Renderer::~Renderer()
{
    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
    }

    if (imguiInitialized)
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        imguiInitialized = false;
    }

    if (xr)
    {
        // Destroy our XR framebuffers/views (which reference runtime swapchain
        // images) BEFORE telling OpenXR to destroy the swapchains/session, and
        // both before the VkDevice goes away below.
        destroyXrRenderTargets();
        if (xrPipeline)
            vkDestroyPipeline(device, xrPipeline, nullptr);
        if (xrRenderPass)
            vkDestroyRenderPass(device, xrRenderPass, nullptr);
        for (size_t i = 0; i < xrUniformBuffers.size(); ++i)
        {
            vkDestroyBuffer(device, xrUniformBuffers[i], nullptr);
            vkFreeMemory(device, xrUniformBuffersMemory[i], nullptr);
        }
        if (xrDescriptorPool)
            vkDestroyDescriptorPool(device, xrDescriptorPool, nullptr);
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
            if (xrInFlightFences[i])
                vkDestroyFence(device, xrInFlightFences[i], nullptr);
        xr->shutdownGraphics();
    }

    // Extra mesh slots.
    for (auto &es : extraSlots)
    {
        if (es.descPool)
            vkDestroyDescriptorPool(device, es.descPool, nullptr);
        for (size_t i = 0; i < es.ubos.size(); ++i)
        {
            vkDestroyBuffer(device, es.ubos[i], nullptr);
            vkFreeMemory(device, es.ubosMem[i], nullptr);
        }
        for (size_t i = 0; i < es.xrUbos.size(); ++i)
        {
            vkDestroyBuffer(device, es.xrUbos[i], nullptr);
            vkFreeMemory(device, es.xrUbosMem[i], nullptr);
        }
        for (auto &t : es.pbr)
            destroyTexture(t);
        if (es.ib)
            vkDestroyBuffer(device, es.ib, nullptr);
        if (es.ibMem)
            vkFreeMemory(device, es.ibMem, nullptr);
        if (es.vb)
            vkDestroyBuffer(device, es.vb, nullptr);
        if (es.vbMem)
            vkFreeMemory(device, es.vbMem, nullptr);
    }
    extraSlots.clear();

    cleanupSwapchain();

    if (graphicsPipeline)
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
    if (pipelineLayout)
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (renderPass)
        vkDestroyRenderPass(device, renderPass, nullptr);

    for (size_t i = 0; i < uniformBuffers.size(); ++i)
    {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }

    for (auto &tex : pbrTextures)
        destroyTexture(tex);
    if (textureSampler)
        vkDestroySampler(device, textureSampler, nullptr);

    if (descriptorPool)
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (descriptorSetLayout)
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    if (indexBuffer)
        vkDestroyBuffer(device, indexBuffer, nullptr);
    if (indexBufferMemory)
        vkFreeMemory(device, indexBufferMemory, nullptr);
    if (vertexBuffer)
        vkDestroyBuffer(device, vertexBuffer, nullptr);
    if (vertexBufferMemory)
        vkFreeMemory(device, vertexBufferMemory, nullptr);

    for (auto s : renderFinishedSemaphores)
        vkDestroySemaphore(device, s, nullptr);
    for (auto s : imageAvailableSemaphores)
        vkDestroySemaphore(device, s, nullptr);
    for (auto f : inFlightFences)
        vkDestroyFence(device, f, nullptr);

    if (commandPool)
        vkDestroyCommandPool(device, commandPool, nullptr);

    if (device)
        vkDestroyDevice(device, nullptr);
    if (surface)
        vkDestroySurfaceKHR(instance, surface, nullptr);
    if (instance)
        vkDestroyInstance(instance, nullptr);
}

void Renderer::createInstance()
{
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "ForceUnleashedVR";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "None";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    uint32_t sdlExtCount = 0;
    const char *const *sdlExts =
        SDL_Vulkan_GetInstanceExtensions(&sdlExtCount);
    if (!sdlExts)
    {
        throw std::runtime_error(std::string("SDL_Vulkan_GetInstanceExtensions: ") +
                                 SDL_GetError());
    }
    std::vector<const char *> extensions(sdlExts, sdlExts + sdlExtCount);

#ifndef NDEBUG
    validationEnabled = validationLayerAvailable();
    if (!validationEnabled)
    {
        SDL_Log("Validation layer not available; continuing without it. "
                "(source the Vulkan SDK setup-env.sh to enable it)");
    }
#endif

    std::vector<const char *> layers;
    if (validationEnabled)
    {
        layers.push_back(kValidationLayer);
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

    if (xr)
    {
        // OpenXR creates the instance, injecting the extensions the runtime
        // needs on top of ours (SDL surface ext + validation layers).
        instance = xr->createVulkanInstance(createInfo);
    }
    else
    {
        vkCheck(vkCreateInstance(&createInfo, nullptr, &instance),
                "vkCreateInstance");
    }
}

void Renderer::createSurface()
{
    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface))
    {
        throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface: ") +
                                 SDL_GetError());
    }
}

void Renderer::pickPhysicalDevice()
{
    std::vector<VkPhysicalDevice> devices;
    if (xr)
    {
        // OpenXR dictates which GPU drives the HMD; we must use exactly that one.
        devices.push_back(xr->getVulkanPhysicalDevice(instance));
    }
    else
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (count == 0)
        {
            throw std::runtime_error("No Vulkan-capable GPU found");
        }
        devices.resize(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
    }

    for (auto dev : devices)
    {
        // Find graphics + present queue families.
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, qprops.data());

        bool foundGraphics = false, foundPresent = false;
        uint32_t gFam = 0, pFam = 0;
        for (uint32_t i = 0; i < qCount; ++i)
        {
            if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                gFam = i;
                foundGraphics = true;
            }
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &present);
            if (present)
            {
                pFam = i;
                foundPresent = true;
            }
        }

        // Check swapchain extension support.
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount,
                                             exts.data());
        bool hasSwapchain = false;
        for (const auto &e : exts)
        {
            if (std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) ==
                0)
            {
                hasSwapchain = true;
                break;
            }
        }

        if (foundGraphics && foundPresent && hasSwapchain)
        {
            physicalDevice = dev;
            graphicsFamily = gFam;
            presentFamily = pFam;

            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);
            SDL_Log("Using GPU: %s", props.deviceName);
            return;
        }
    }

    throw std::runtime_error("No suitable GPU (graphics+present+swapchain)");
}

void Renderer::createLogicalDevice()
{
    std::set<uint32_t> uniqueFamilies{graphicsFamily, presentFamily};
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    float priority = 1.0f;
    for (uint32_t fam : uniqueFamilies)
    {
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = fam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        queueInfos.push_back(qi);
    }

    VkPhysicalDeviceFeatures features{};

    const char *deviceExts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos = queueInfos.data();
    createInfo.pEnabledFeatures = &features;
    createInfo.enabledExtensionCount = 1;
    createInfo.ppEnabledExtensionNames = deviceExts;

    if (xr)
    {
        // OpenXR creates the device, injecting any extensions the runtime needs.
        device = xr->createVulkanDevice(physicalDevice, createInfo);
    }
    else
    {
        vkCheck(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device),
                "vkCreateDevice");
    }

    // OpenXR's graphics binding uses queueIndex 0 of graphicsFamily — match it.
    vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, presentFamily, 0, &presentQueue);
}

void Renderer::createSwapchain()
{
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &caps);

    // Surface format: prefer B8G8R8A8_SRGB.
    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &fmtCount,
                                         nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &fmtCount,
                                         formats.data());
    VkSurfaceFormatKHR chosen = formats[0];
    for (const auto &f : formats)
    {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            chosen = f;
            break;
        }
    }
    swapchainFormat = chosen.format;

    // Extent.
    if (caps.currentExtent.width != UINT32_MAX)
    {
        swapchainExtent = caps.currentExtent;
    }
    else
    {
        int w = 0, h = 0;
        SDL_GetWindowSizeInPixels(window, &w, &h);
        swapchainExtent.width =
            std::clamp(static_cast<uint32_t>(w), caps.minImageExtent.width,
                       caps.maxImageExtent.width);
        swapchainExtent.height =
            std::clamp(static_cast<uint32_t>(h), caps.minImageExtent.height,
                       caps.maxImageExtent.height);
    }

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
    {
        imageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = swapchainFormat;
    createInfo.imageColorSpace = chosen.colorSpace;
    createInfo.imageExtent = swapchainExtent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t famIndices[] = {graphicsFamily, presentFamily};
    if (graphicsFamily != presentFamily)
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = famIndices;
    }
    else
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = caps.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    vkCheck(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain),
            "vkCreateSwapchainKHR");

    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount,
                            swapchainImages.data());
}

VkImageView Renderer::createImageView(VkImage image, VkFormat format,
                                      VkImageAspectFlags aspect,
                                      uint32_t mipLevels)
{
    VkImageViewCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.subresourceRange.aspectMask = aspect;
    info.subresourceRange.levelCount = mipLevels;
    info.subresourceRange.layerCount = 1;

    VkImageView view;
    vkCheck(vkCreateImageView(device, &info, nullptr, &view),
            "vkCreateImageView");
    return view;
}

void Renderer::createImageViews()
{
    swapchainImageViews.resize(swapchainImages.size());
    for (size_t i = 0; i < swapchainImages.size(); ++i)
    {
        swapchainImageViews[i] = createImageView(
            swapchainImages[i], swapchainFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }
}

void Renderer::createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
                           VkFormat format, VkImageTiling tiling,
                           VkImageUsageFlags usage,
                           VkMemoryPropertyFlags props, VkImage &image,
                           VkDeviceMemory &memory,
                           VkSampleCountFlagBits samples)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = samples;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkCheck(vkCreateImage(device, &imageInfo, nullptr, &image),
            "vkCreateImage");

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, image, &memReq);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = memReq.size;
    alloc.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, props);
    vkCheck(vkAllocateMemory(device, &alloc, nullptr, &memory),
            "vkAllocateMemory (image)");
    vkBindImageMemory(device, image, memory, 0);
}

void Renderer::createMsaaColorResources()
{
    createImage(swapchainExtent.width, swapchainExtent.height, 1,
                swapchainFormat, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, msaaColor.image,
                msaaColor.memory, msaaSamples);
    msaaColor.view = createImageView(msaaColor.image, swapchainFormat,
                                     VK_IMAGE_ASPECT_COLOR_BIT, 1);
    msaaColor.mipLevels = 1;
}

void Renderer::createDepthResources()
{
    createImage(swapchainExtent.width, swapchainExtent.height, 1, depthFormat,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage,
                depthImageMemory, msaaSamples);
    depthImageView =
        createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
}

void Renderer::createRenderPass()
{
    // Attachment 0: MSAA color (multisampled, not directly presentable)
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapchainFormat;
    colorAttachment.samples = msaaSamples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Attachment 1: MSAA depth
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = depthFormat;
    depthAttachment.samples = msaaSamples;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Attachment 2: Resolve target (single-sample swapchain image)
    VkAttachmentDescription resolveAttachment{};
    resolveAttachment.format = swapchainFormat;
    resolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    resolveAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolveAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    resolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    resolveAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    resolveAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 1;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference resolveRef{};
    resolveRef.attachment = 2;
    resolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;
    subpass.pResolveAttachments = &resolveRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask =
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 3> attachments = {
        colorAttachment, depthAttachment, resolveAttachment};
    VkRenderPassCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = static_cast<uint32_t>(attachments.size());
    info.pAttachments = attachments.data();
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 1;
    info.pDependencies = &dependency;

    vkCheck(vkCreateRenderPass(device, &info, nullptr, &renderPass),
            "vkCreateRenderPass");
}

void Renderer::createDescriptorSetLayout()
{
    // Binding 0: UBO. Bindings 1-4: base/normal/roughness/metallic samplers.
    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    for (uint32_t i = 1; i < 5; ++i)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = static_cast<uint32_t>(bindings.size());
    info.pBindings = bindings.data();

    vkCheck(vkCreateDescriptorSetLayout(device, &info, nullptr,
                                        &descriptorSetLayout),
            "vkCreateDescriptorSetLayout");
}

VkShaderModule Renderer::createShaderModule(const char *spvPath)
{
    size_t size = 0;
    char *code = read_binary_file(spvPath, &size);
    if (!code || size == 0)
    {
        throw std::runtime_error(std::string("Failed to read shader: ") +
                                 spvPath);
    }

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode = reinterpret_cast<const uint32_t *>(code);

    VkShaderModule module;
    VkResult res = vkCreateShaderModule(device, &info, nullptr, &module);
    free(code);
    vkCheck(res, "vkCreateShaderModule");
    return module;
}

void Renderer::createGraphicsPipeline()
{
    // Pipeline layout is shared by the desktop and XR pipelines.
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout;
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(PushConstants);
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    vkCheck(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout),
            "vkCreatePipelineLayout");

    graphicsPipeline = buildPipeline(renderPass, msaaSamples);
}

VkPipeline Renderer::buildPipeline(VkRenderPass rp,
                                   VkSampleCountFlagBits samples)
{
    VkShaderModule vert = createShaderModule(SHADER_DIR "/model.vert.spv");
    VkShaderModule frag = createShaderModule(SHADER_DIR "/model.frag.spv");

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vert;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = frag;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(PackedVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    // Quantized formats; Vulkan converts unorm/snorm/half to float at fetch.
    std::array<VkVertexInputAttributeDescription, 4> attrs{};
    attrs[0].location = 0;
    attrs[0].binding = 0;
    attrs[0].format = VK_FORMAT_R16G16B16A16_UNORM;
    attrs[0].offset = offsetof(PackedVertex, pos);
    attrs[1].location = 1;
    attrs[1].binding = 0;
    attrs[1].format = VK_FORMAT_R8G8B8A8_SNORM;
    attrs[1].offset = offsetof(PackedVertex, normal);
    attrs[2].location = 2;
    attrs[2].binding = 0;
    attrs[2].format = VK_FORMAT_R8G8B8A8_SNORM;
    attrs[2].offset = offsetof(PackedVertex, tangent);
    attrs[3].location = 3;
    attrs[3].binding = 0;
    attrs[3].format = VK_FORMAT_R16G16_SFLOAT;
    attrs[3].offset = offsetof(PackedVertex, uv);

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attrs.size());
    vertexInput.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth = 1.0f;
    // Backface culling disabled: model formats (notably FBX via
    // PreTransformVertices) bake mirrored transforms that flip winding per-mesh,
    // so culling would hide near faces. Lighting is made two-sided in the
    // fragment shader (gl_FrontFacing) instead.
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = samples;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &colorBlendAttachment;

    std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                               VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
    dynamicState.pDynamicStates = dynStates.data();

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = rp;
    pipelineInfo.subpass = 0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCheck(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                      nullptr, &pipeline),
            "vkCreateGraphicsPipelines");

    vkDestroyShaderModule(device, frag, nullptr);
    vkDestroyShaderModule(device, vert, nullptr);
    return pipeline;
}

void Renderer::createFramebuffers()
{
    framebuffers.resize(swapchainImageViews.size());
    for (size_t i = 0; i < swapchainImageViews.size(); ++i)
    {
        std::array<VkImageView, 3> attachments = {
            msaaColor.view, depthImageView, swapchainImageViews[i]};
        VkFramebufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.renderPass = renderPass;
        info.attachmentCount = static_cast<uint32_t>(attachments.size());
        info.pAttachments = attachments.data();
        info.width = swapchainExtent.width;
        info.height = swapchainExtent.height;
        info.layers = 1;
        vkCheck(vkCreateFramebuffer(device, &info, nullptr, &framebuffers[i]),
                "vkCreateFramebuffer");
    }
}

void Renderer::createCommandPool()
{
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = graphicsFamily;
    vkCheck(vkCreateCommandPool(device, &info, nullptr, &commandPool),
            "vkCreateCommandPool");
}

uint32_t Renderer::findMemoryType(uint32_t typeFilter,
                                  VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
    {
        if ((typeFilter & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props)
        {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags props, VkBuffer &buffer,
                            VkDeviceMemory &memory)
{
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateBuffer(device, &info, nullptr, &buffer), "vkCreateBuffer");

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = memReq.size;
    alloc.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, props);
    vkCheck(vkAllocateMemory(device, &alloc, nullptr, &memory),
            "vkAllocateMemory (buffer)");
    vkBindBufferMemory(device, buffer, memory, 0);
}

VkCommandBuffer Renderer::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    return cmd;
}

void Renderer::endSingleTimeCommands(VkCommandBuffer cmd)
{
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

void Renderer::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
{
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    endSingleTimeCommands(cmd);
}

void Renderer::transitionImageLayout(VkImage image, VkImageLayout oldLayout,
                                     VkImageLayout newLayout,
                                     uint32_t mipLevels)
{
    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        throw std::runtime_error("Unsupported image layout transition");
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);
    endSingleTimeCommands(cmd);
}

void Renderer::copyBufferToImage(VkBuffer buffer, VkImage image,
                                 uint32_t width, uint32_t height)
{
    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(cmd, buffer, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    endSingleTimeCommands(cmd);
}

void Renderer::generateMipmaps(VkImage image, VkFormat format, int32_t width,
                               int32_t height, uint32_t mipLevels)
{
    // Verify the format supports linear blitting.
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
    if (!(props.optimalTilingFeatures &
          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
    {
        // No linear blit: just transition the single (already-filled) level 0
        // and leave the rest. Treat all levels as shader-read.
        transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              mipLevels);
        return;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    int32_t mipW = width;
    int32_t mipH = height;

    for (uint32_t i = 1; i < mipLevels; ++i)
    {
        // Transition level i-1 to transfer-src for blitting down.
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[1] = {mipW, mipH, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[1] = {mipW > 1 ? mipW / 2 : 1, mipH > 1 ? mipH / 2 : 1,
                              1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.layerCount = 1;
        vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                       VK_FILTER_LINEAR);

        // Level i-1 is done: move it to shader-read.
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);

        if (mipW > 1)
            mipW /= 2;
        if (mipH > 1)
            mipH /= 2;
    }

    // Last mip level: still in transfer-dst, move to shader-read.
    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);

    endSingleTimeCommands(cmd);
}

Renderer::Texture Renderer::createSolidTexture(const uint8_t rgba[4],
                                               VkFormat format)
{
    Texture tex;
    tex.mipLevels = 1;

    VkDeviceSize imageSize = 4;
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);
    void *data;
    vkMapMemory(device, stagingMem, 0, imageSize, 0, &data);
    std::memcpy(data, rgba, 4);
    vkUnmapMemory(device, stagingMem);

    createImage(1, 1, 1, format, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tex.image, tex.memory);

    transitionImageLayout(tex.image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1);
    copyBufferToImage(staging, tex.image, 1, 1);
    transitionImageLayout(tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    tex.view =
        createImageView(tex.image, format, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    return tex;
}

Renderer::Texture Renderer::loadOrDefaultTexture(const std::string &path,
                                                 VkFormat format,
                                                 const uint8_t fallback[4])
{
    if (path.empty())
    {
        return createSolidTexture(fallback, format);
    }

    int width = 0, height = 0, channels = 0;
    stbi_uc *pixels =
        stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels)
    {
        SDL_Log("Failed to load texture '%s' (%s); using default", path.c_str(),
                stbi_failure_reason());
        return createSolidTexture(fallback, format);
    }

    Texture tex;
    tex.mipLevels = static_cast<uint32_t>(
                        std::floor(std::log2(std::max(width, height)))) +
                    1;

    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);
    void *data;
    vkMapMemory(device, stagingMem, 0, imageSize, 0, &data);
    std::memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingMem);
    stbi_image_free(pixels);

    createImage(width, height, tex.mipLevels, format, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tex.image, tex.memory);

    transitionImageLayout(tex.image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, tex.mipLevels);
    copyBufferToImage(staging, tex.image, width, height);
    // generateMipmaps transitions all levels to SHADER_READ_ONLY_OPTIMAL.
    generateMipmaps(tex.image, format, width, height, tex.mipLevels);

    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    tex.view = createImageView(tex.image, format, VK_IMAGE_ASPECT_COLOR_BIT,
                               tex.mipLevels);
    return tex;
}

void Renderer::destroyTexture(Texture &tex)
{
    if (tex.view)
        vkDestroyImageView(device, tex.view, nullptr);
    if (tex.image)
        vkDestroyImage(device, tex.image, nullptr);
    if (tex.memory)
        vkFreeMemory(device, tex.memory, nullptr);
    tex.view = VK_NULL_HANDLE;
    tex.image = VK_NULL_HANDLE;
    tex.memory = VK_NULL_HANDLE;
}

void Renderer::createTextureSampler()
{
    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = VK_FILTER_LINEAR;
    info.minFilter = VK_FILTER_LINEAR;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.anisotropyEnable = VK_FALSE;
    info.maxAnisotropy = 1.0f;
    info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    info.unnormalizedCoordinates = VK_FALSE;
    info.compareEnable = VK_FALSE;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    info.minLod = 0.0f;
    info.maxLod = VK_LOD_CLAMP_NONE;

    vkCheck(vkCreateSampler(device, &info, nullptr, &textureSampler),
            "vkCreateSampler");
}

void Renderer::createMeshBuffers(const Model &model)
{
    const std::vector<Vertex> &vertices = model.vertices;
    const std::vector<uint32_t> &indices = model.indices;
    indexCount = static_cast<uint32_t>(indices.size());

    // Destroy any previously-uploaded mesh (LOD change / re-upload).
    if (vertexBuffer)
        vkDestroyBuffer(device, vertexBuffer, nullptr);
    if (vertexBufferMemory)
        vkFreeMemory(device, vertexBufferMemory, nullptr);
    if (indexBuffer)
        vkDestroyBuffer(device, indexBuffer, nullptr);
    if (indexBufferMemory)
        vkFreeMemory(device, indexBufferMemory, nullptr);
    vertexBuffer = VK_NULL_HANDLE;
    vertexBufferMemory = VK_NULL_HANDLE;
    indexBuffer = VK_NULL_HANDLE;
    indexBufferMemory = VK_NULL_HANDLE;

    // Quantize float vertices into the compact PackedVertex layout. Position is
    // normalized into the stored bbox so the shader can dequantize uniformly.
    std::vector<PackedVertex> packed(vertices.size());
    const glm::vec3 invExtent = {
        quantExtent.x > 0.0f ? 1.0f / quantExtent.x : 0.0f,
        quantExtent.y > 0.0f ? 1.0f / quantExtent.y : 0.0f,
        quantExtent.z > 0.0f ? 1.0f / quantExtent.z : 0.0f};
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        const Vertex &v = vertices[i];
        PackedVertex &p = packed[i];

        glm::vec3 n = (v.pos - quantMin) * invExtent; // -> [0,1]
        p.pos[0] = static_cast<uint16_t>(meshopt_quantizeUnorm(n.x, 16));
        p.pos[1] = static_cast<uint16_t>(meshopt_quantizeUnorm(n.y, 16));
        p.pos[2] = static_cast<uint16_t>(meshopt_quantizeUnorm(n.z, 16));
        p.pos[3] = 0;

        p.normal[0] = static_cast<int8_t>(meshopt_quantizeSnorm(v.normal.x, 8));
        p.normal[1] = static_cast<int8_t>(meshopt_quantizeSnorm(v.normal.y, 8));
        p.normal[2] = static_cast<int8_t>(meshopt_quantizeSnorm(v.normal.z, 8));
        p.normal[3] = 0;

        p.tangent[0] = static_cast<int8_t>(meshopt_quantizeSnorm(v.tangent.x, 8));
        p.tangent[1] = static_cast<int8_t>(meshopt_quantizeSnorm(v.tangent.y, 8));
        p.tangent[2] = static_cast<int8_t>(meshopt_quantizeSnorm(v.tangent.z, 8));
        p.tangent[3] = static_cast<int8_t>(meshopt_quantizeSnorm(v.tangent.w, 8));

        p.uv[0] = meshopt_quantizeHalf(v.uv.x);
        p.uv[1] = meshopt_quantizeHalf(v.uv.y);
    }

    // Vertex buffer.
    {
        VkDeviceSize size = sizeof(PackedVertex) * packed.size();
        VkBuffer staging;
        VkDeviceMemory stagingMem;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMem);
        void *data;
        vkMapMemory(device, stagingMem, 0, size, 0, &data);
        std::memcpy(data, packed.data(), static_cast<size_t>(size));
        vkUnmapMemory(device, stagingMem);

        createBuffer(size,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer,
                     vertexBufferMemory);
        copyBuffer(staging, vertexBuffer, size);

        vkDestroyBuffer(device, staging, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);
    }

    // Index buffer.
    {
        VkDeviceSize size = sizeof(uint32_t) * indices.size();
        VkBuffer staging;
        VkDeviceMemory stagingMem;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMem);
        void *data;
        vkMapMemory(device, stagingMem, 0, size, 0, &data);
        std::memcpy(data, indices.data(), static_cast<size_t>(size));
        vkUnmapMemory(device, stagingMem);

        createBuffer(size,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer,
                     indexBufferMemory);
        copyBuffer(staging, indexBuffer, size);

        vkDestroyBuffer(device, staging, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);
    }
}

void Renderer::updateMesh(const Model &model)
{
    vkDeviceWaitIdle(device);
    createMeshBuffers(model);
}

void Renderer::uploadObject(const Model &model, const TextureSet &textures)
{
    // Compute the position-dequantization bbox from this full-res model and
    // keep it for all later updateMesh() calls (simplification never grows it).
    glm::vec3 minB(std::numeric_limits<float>::max());
    glm::vec3 maxB(std::numeric_limits<float>::lowest());
    for (const Vertex &v : model.vertices)
    {
        minB = glm::min(minB, v.pos);
        maxB = glm::max(maxB, v.pos);
    }
    if (model.vertices.empty())
    {
        minB = glm::vec3(0.0f);
        maxB = glm::vec3(1.0f);
    }
    quantMin = minB;
    quantExtent = maxB - minB;
    // Guard against a zero extent on any axis (flat meshes).
    if (quantExtent.x <= 0.0f)
        quantExtent.x = 1.0f;
    if (quantExtent.y <= 0.0f)
        quantExtent.y = 1.0f;
    if (quantExtent.z <= 0.0f)
        quantExtent.z = 1.0f;

    createMeshBuffers(model);

    // PBR textures. base is sRGB (color); the others are linear data. Empty
    // paths fall back to neutral 1x1 constants so plain models still render.
    const uint8_t whiteRGBA[4] = {255, 255, 255, 255};
    const uint8_t flatNormalRGBA[4] = {128, 128, 255, 255};
    const uint8_t roughRGBA[4] = {255, 255, 255, 255}; // roughness = 1
    const uint8_t metalRGBA[4] = {0, 0, 0, 255};       // metallic = 0

    pbrTextures[0] = loadOrDefaultTexture(textures.base,
                                          VK_FORMAT_R8G8B8A8_SRGB, whiteRGBA);
    pbrTextures[1] = loadOrDefaultTexture(
        textures.normal, VK_FORMAT_R8G8B8A8_UNORM, flatNormalRGBA);
    pbrTextures[2] = loadOrDefaultTexture(
        textures.roughness, VK_FORMAT_R8G8B8A8_UNORM, roughRGBA);
    pbrTextures[3] = loadOrDefaultTexture(
        textures.metallic, VK_FORMAT_R8G8B8A8_UNORM, metalRGBA);

    // Descriptors can only be written now that the textures exist.
    createDescriptorSets();
}

uint32_t Renderer::uploadMesh(const Model &model, const TextureSet &textures)
{
    // Validate device is ready (uploadObject must have been called first).
    if (device == VK_NULL_HANDLE || textureSampler == VK_NULL_HANDLE)
        throw std::runtime_error("uploadMesh called before renderer is fully initialised");

    ExtraSlot es;

    // Compute bbox for this mesh's dequantization.
    glm::vec3 minB(std::numeric_limits<float>::max());
    glm::vec3 maxB(std::numeric_limits<float>::lowest());
    for (const Vertex &v : model.vertices)
    {
        minB = glm::min(minB, v.pos);
        maxB = glm::max(maxB, v.pos);
    }
    if (model.vertices.empty())
    {
        minB = glm::vec3(0.0f);
        maxB = glm::vec3(1.0f);
    }
    es.quantMin = minB;
    es.quantExtent = maxB - minB;
    if (es.quantExtent.x <= 0.0f)
        es.quantExtent.x = 1.0f;
    if (es.quantExtent.y <= 0.0f)
        es.quantExtent.y = 1.0f;
    if (es.quantExtent.z <= 0.0f)
        es.quantExtent.z = 1.0f;

    // Upload vertex/index buffers (same quantization path as slot 0).
    {
        // Temporarily swap the global quant fields so createMeshBuffers can run.
        glm::vec3 savedMin = quantMin, savedExt = quantExtent;
        quantMin = es.quantMin;
        quantExtent = es.quantExtent;
        // We need separate buffers; save slot-0 buffers, point globals to nulls.
        VkBuffer savedVB = vertexBuffer, savedIB = indexBuffer;
        VkDeviceMemory savedVBMem = vertexBufferMemory, savedIBMem = indexBufferMemory;
        uint32_t savedIC = indexCount;
        vertexBuffer = VK_NULL_HANDLE;
        indexBuffer = VK_NULL_HANDLE;
        vertexBufferMemory = VK_NULL_HANDLE;
        indexBufferMemory = VK_NULL_HANDLE;
        createMeshBuffers(model); // creates new vertex/index buffers in globals
        es.vb = vertexBuffer;
        es.vbMem = vertexBufferMemory;
        es.ib = indexBuffer;
        es.ibMem = indexBufferMemory;
        es.indexCount = indexCount;
        // Restore slot-0 buffers.
        vertexBuffer = savedVB;
        indexBuffer = savedIB;
        vertexBufferMemory = savedVBMem;
        indexBufferMemory = savedIBMem;
        indexCount = savedIC;
        quantMin = savedMin;
        quantExtent = savedExt;
    }

    // Load PBR textures.
    const uint8_t white[4] = {255, 255, 255, 255};
    const uint8_t flatN[4] = {128, 128, 255, 255};
    const uint8_t rough[4] = {255, 255, 255, 255};
    const uint8_t metal[4] = {0, 0, 0, 255};
    es.pbr[0] = loadOrDefaultTexture(textures.base, VK_FORMAT_R8G8B8A8_SRGB, white);
    es.pbr[1] = loadOrDefaultTexture(textures.normal, VK_FORMAT_R8G8B8A8_UNORM, flatN);
    es.pbr[2] = loadOrDefaultTexture(textures.roughness, VK_FORMAT_R8G8B8A8_UNORM, rough);
    es.pbr[3] = loadOrDefaultTexture(textures.metallic, VK_FORMAT_R8G8B8A8_UNORM, metal);

    // Create per-slot UBOs (desktop + XR).
    const uint32_t xrSlots = xr ? kXrSlots : 0u;
    const uint32_t totalUboSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) + xrSlots;
    const VkDeviceSize uboSize = sizeof(UniformBufferObject);
    auto allocUbos = [&](uint32_t count,
                         std::vector<VkBuffer> &bufs,
                         std::vector<VkDeviceMemory> &mems,
                         std::vector<void *> &maps)
    {
        bufs.resize(count);
        mems.resize(count);
        maps.resize(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            createBuffer(uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         bufs[i], mems[i]);
            vkMapMemory(device, mems[i], 0, uboSize, 0, &maps[i]);
        }
    };
    allocUbos(MAX_FRAMES_IN_FLIGHT, es.ubos, es.ubosMem, es.ubosMapped);
    if (xrSlots)
        allocUbos(xrSlots, es.xrUbos, es.xrUbosMem, es.xrUbosMapped);

    // Create descriptor pool + sets for this slot.
    {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, totalUboSets};
        poolSizes[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, totalUboSets * 4};
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        pi.pPoolSizes = poolSizes.data();
        pi.maxSets = totalUboSets;
        vkCheck(vkCreateDescriptorPool(device, &pi, nullptr, &es.descPool),
                "vkCreateDescriptorPool (extra slot)");

        auto allocSets = [&](std::vector<VkDescriptorSet> &sets,
                             const std::vector<VkBuffer> &ubos,
                             uint32_t count)
        {
            std::vector<VkDescriptorSetLayout> layouts(count, descriptorSetLayout);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = es.descPool;
            ai.descriptorSetCount = count;
            ai.pSetLayouts = layouts.data();
            sets.resize(count);
            vkCheck(vkAllocateDescriptorSets(device, &ai, sets.data()),
                    "vkAllocateDescriptorSets (extra slot)");
            for (uint32_t i = 0; i < count; ++i)
            {
                VkDescriptorBufferInfo bi{};
                bi.buffer = ubos[i];
                bi.offset = 0;
                bi.range = uboSize;
                std::array<VkDescriptorImageInfo, 4> ii{};
                for (int t = 0; t < 4; ++t)
                {
                    ii[t].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    ii[t].imageView = es.pbr[t].view;
                    ii[t].sampler = textureSampler;
                }
                std::array<VkWriteDescriptorSet, 5> wr{};
                wr[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                         sets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bi};
                for (int t = 0; t < 4; ++t)
                    wr[t + 1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                                 sets[i], static_cast<uint32_t>(t + 1), 0, 1,
                                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii[t]};
                vkUpdateDescriptorSets(device, static_cast<uint32_t>(wr.size()), wr.data(), 0, nullptr);
            }
        };
        allocSets(es.descSets, es.ubos, MAX_FRAMES_IN_FLIGHT);
        if (xrSlots)
            allocSets(es.xrDescSets, es.xrUbos, xrSlots);
    }

    extraSlots.push_back(std::move(es));
    return static_cast<uint32_t>(extraSlots.size()); // meshId = 1-based index
}

void Renderer::createUniformBuffers()
{
    VkDeviceSize size = sizeof(UniformBufferObject);
    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     uniformBuffers[i], uniformBuffersMemory[i]);
        vkMapMemory(device, uniformBuffersMemory[i], 0, size, 0,
                    &uniformBuffersMapped[i]);
    }
}

void Renderer::createDescriptorPool()
{
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = MAX_FRAMES_IN_FLIGHT;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = MAX_FRAMES_IN_FLIGHT * 4; // 4 maps per frame

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    info.pPoolSizes = poolSizes.data();
    info.maxSets = MAX_FRAMES_IN_FLIGHT;
    vkCheck(vkCreateDescriptorPool(device, &info, nullptr, &descriptorPool),
            "vkCreateDescriptorPool");
}

void Renderer::createDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                               descriptorSetLayout);
    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = descriptorPool;
    alloc.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    alloc.pSetLayouts = layouts.data();

    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    vkCheck(vkAllocateDescriptorSets(device, &alloc, descriptorSets.data()),
            "vkAllocateDescriptorSets");

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        std::array<VkDescriptorImageInfo, 4> imageInfos{};
        for (int t = 0; t < 4; ++t)
        {
            imageInfos[t].imageLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfos[t].imageView = pbrTextures[t].view;
            imageInfos[t].sampler = textureSampler;
        }

        std::array<VkWriteDescriptorSet, 5> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSets[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &bufferInfo;

        for (int t = 0; t < 4; ++t)
        {
            writes[t + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[t + 1].dstSet = descriptorSets[i];
            writes[t + 1].dstBinding = t + 1;
            writes[t + 1].descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[t + 1].descriptorCount = 1;
            writes[t + 1].pImageInfo = &imageInfos[t];
        }

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    // XR per-eye descriptor sets (allocated up front in createXrFrameResources);
    // each points at its own UBO slot and the shared PBR textures.
    if (xr)
    {
        for (uint32_t s = 0; s < kXrSlots; ++s)
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = xrUniformBuffers[s];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            std::array<VkDescriptorImageInfo, 4> imageInfos{};
            for (int t = 0; t < 4; ++t)
            {
                imageInfos[t].imageLayout =
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfos[t].imageView = pbrTextures[t].view;
                imageInfos[t].sampler = textureSampler;
            }

            std::array<VkWriteDescriptorSet, 5> writes{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = xrDescriptorSets[s];
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].descriptorCount = 1;
            writes[0].pBufferInfo = &bufferInfo;

            for (int t = 0; t < 4; ++t)
            {
                writes[t + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[t + 1].dstSet = xrDescriptorSets[s];
                writes[t + 1].dstBinding = t + 1;
                writes[t + 1].descriptorType =
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[t + 1].descriptorCount = 1;
                writes[t + 1].pImageInfo = &imageInfos[t];
            }
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
        }
    }
}

void Renderer::createCommandBuffers()
{
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = commandPool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    vkCheck(vkAllocateCommandBuffers(device, &info, commandBuffers.data()),
            "vkAllocateCommandBuffers");
}

void Renderer::createSyncObjects()
{
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    // One render-finished semaphore per swapchain image avoids reusing a
    // semaphore that a pending present might still be waiting on.
    renderFinishedSemaphores.resize(swapchainImages.size());

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vkCheck(vkCreateSemaphore(device, &semInfo, nullptr,
                                  &imageAvailableSemaphores[i]),
                "vkCreateSemaphore (imageAvailable)");
        vkCheck(vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]),
                "vkCreateFence");
    }
    for (size_t i = 0; i < renderFinishedSemaphores.size(); ++i)
    {
        vkCheck(vkCreateSemaphore(device, &semInfo, nullptr,
                                  &renderFinishedSemaphores[i]),
                "vkCreateSemaphore (renderFinished)");
    }
}

void Renderer::cleanupSwapchain()
{
    for (auto fb : framebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    framebuffers.clear();

    destroyTexture(msaaColor);

    if (depthImageView)
        vkDestroyImageView(device, depthImageView, nullptr);
    if (depthImage)
        vkDestroyImage(device, depthImage, nullptr);
    if (depthImageMemory)
        vkFreeMemory(device, depthImageMemory, nullptr);
    depthImageView = VK_NULL_HANDLE;
    depthImage = VK_NULL_HANDLE;
    depthImageMemory = VK_NULL_HANDLE;

    for (auto v : swapchainImageViews)
        vkDestroyImageView(device, v, nullptr);
    swapchainImageViews.clear();

    if (swapchain)
        vkDestroySwapchainKHR(device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
}

void Renderer::recreateSwapchain()
{
    // Wait until the window has a non-zero size (e.g. un-minimized).
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window, &w, &h);
    while (w == 0 || h == 0)
    {
        SDL_GetWindowSizeInPixels(window, &w, &h);
        SDL_WaitEvent(nullptr);
    }

    vkDeviceWaitIdle(device);

    cleanupSwapchain();

    createSwapchain();
    createImageViews();
    createMsaaColorResources();
    createDepthResources();
    createFramebuffers();

    if (imguiInitialized)
    {
        ImGui_ImplVulkan_SetMinImageCount(
            static_cast<uint32_t>(swapchainImages.size()));
    }

    // Number of swapchain images can change; resize render-finished semaphores.
    for (auto s : renderFinishedSemaphores)
        vkDestroySemaphore(device, s, nullptr);
    renderFinishedSemaphores.resize(swapchainImages.size());
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (size_t i = 0; i < renderFinishedSemaphores.size(); ++i)
    {
        vkCheck(vkCreateSemaphore(device, &semInfo, nullptr,
                                  &renderFinishedSemaphores[i]),
                "vkCreateSemaphore (renderFinished, recreate)");
    }
}

void Renderer::updateUniformBuffer(uint32_t frame, const glm::mat4 &view,
                                   const glm::mat4 &proj,
                                   const glm::vec3 &lightDir,
                                   const glm::vec3 &camPos)
{
    UniformBufferObject ubo{};
    ubo.view = view;
    ubo.proj = proj;
    ubo.lightDir = glm::vec4(lightDir, 0.0f);
    ubo.camPos = glm::vec4(camPos, 1.0f);
    // Slot 0
    ubo.quantMin = glm::vec4(quantMin, 0.0f);
    ubo.quantExtent = glm::vec4(quantExtent, 0.0f);
    std::memcpy(uniformBuffersMapped[frame], &ubo, sizeof(ubo));
    // Extra slots
    for (auto &es : extraSlots)
    {
        if (frame < es.ubosMapped.size())
        {
            ubo.quantMin = glm::vec4(es.quantMin, 0.0f);
            ubo.quantExtent = glm::vec4(es.quantExtent, 0.0f);
            std::memcpy(es.ubosMapped[frame], &ubo, sizeof(ubo));
        }
    }
}

void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCheck(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");

    // 3 clear values: MSAA color, depth, resolve (resolve loadOp=DONT_CARE so value is ignored)
    std::array<VkClearValue, 3> clears{};
    clears[0].color = {{0.05f, 0.05f, 0.08f, 1.0f}};
    clears[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = renderPass;
    rpBegin.framebuffer = framebuffers[imageIndex];
    rpBegin.renderArea.extent = swapchainExtent;
    rpBegin.clearValueCount = static_cast<uint32_t>(clears.size());
    rpBegin.pClearValues = clears.data();

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport{};
    viewport.width = static_cast<float>(swapchainExtent.width);
    viewport.height = static_cast<float>(swapchainExtent.height);
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = swapchainExtent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VkBuffer vbs[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};

    uint32_t lastMeshId = UINT32_MAX;
    for (const DrawCall &dc : drawCalls)
    {
        const uint32_t mid = dc.meshId;
        // Rebind vertex/index/descriptors only when the mesh slot changes.
        if (mid != lastMeshId)
        {
            if (mid == 0)
            {
                vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, offsets);
                vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayout, 0, 1,
                                        &descriptorSets[currentFrame], 0, nullptr);
            }
            else if (mid - 1 < extraSlots.size())
            {
                auto &es = extraSlots[mid - 1];
                vkCmdBindVertexBuffers(cmd, 0, 1, &es.vb, offsets);
                vkCmdBindIndexBuffer(cmd, es.ib, 0, VK_INDEX_TYPE_UINT32);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayout, 0, 1,
                                        &es.descSets[currentFrame], 0, nullptr);
            }
            lastMeshId = mid;
        }
        const uint32_t ic =
            (mid == 0) ? indexCount
                       : (mid - 1 < extraSlots.size() ? extraSlots[mid - 1].indexCount : 0u);
        if (ic == 0)
            continue;

        PushConstants pc{dc.transform};
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, ic, 1, 0, 0, 0);
    }

    if (ImDrawData *dd = ImGui::GetDrawData())
        ImGui_ImplVulkan_RenderDrawData(dd, cmd);

    vkCmdEndRenderPass(cmd);
    vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
}

VkSampleCountFlagBits Renderer::getMaxMsaaSamples() const
{
    return getMaxSampleCount(physicalDevice, swapchainFormat);
}

void Renderer::setMsaaSamples(VkSampleCountFlagBits samples)
{
    if (samples == msaaSamples)
        return;
    msaaSamples = samples;

    vkDeviceWaitIdle(device);

    if (imguiInitialized)
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        imguiInitialized = false;
    }

    for (auto fb : framebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    framebuffers.clear();

    destroyTexture(msaaColor);

    if (depthImageView)
        vkDestroyImageView(device, depthImageView, nullptr);
    if (depthImage)
        vkDestroyImage(device, depthImage, nullptr);
    if (depthImageMemory)
        vkFreeMemory(device, depthImageMemory, nullptr);
    depthImageView = VK_NULL_HANDLE;
    depthImage = VK_NULL_HANDLE;
    depthImageMemory = VK_NULL_HANDLE;

    if (graphicsPipeline)
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
    if (pipelineLayout)
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (renderPass)
        vkDestroyRenderPass(device, renderPass, nullptr);
    graphicsPipeline = VK_NULL_HANDLE;
    pipelineLayout = VK_NULL_HANDLE;
    renderPass = VK_NULL_HANDLE;

    createMsaaColorResources();
    createDepthResources();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    initImGui();

    // The XR pipeline/render pass also depend on the sample count; rebuild them.
    if (xr)
    {
        destroyXrRenderTargets();
        if (xrPipeline)
            vkDestroyPipeline(device, xrPipeline, nullptr);
        if (xrRenderPass)
            vkDestroyRenderPass(device, xrRenderPass, nullptr);
        xrPipeline = VK_NULL_HANDLE;
        xrRenderPass = VK_NULL_HANDLE;
        createXrRenderPass();
        xrPipeline = buildPipeline(xrRenderPass, msaaSamples);
        createXrRenderTargets();
    }
}

void Renderer::initImGui()
{
    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplSDL3_InitForVulkan(window);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = instance;
    initInfo.PhysicalDevice = physicalDevice;
    initInfo.Device = device;
    initInfo.QueueFamily = graphicsFamily;
    initInfo.Queue = graphicsQueue;
    initInfo.DescriptorPoolSize = 8; // ImGui manages its own pool internally (min 8)
    initInfo.MinImageCount = static_cast<uint32_t>(swapchainImages.size());
    initInfo.ImageCount = static_cast<uint32_t>(swapchainImages.size());
    initInfo.PipelineInfoMain.RenderPass = renderPass;
    initInfo.PipelineInfoMain.MSAASamples = msaaSamples;
    ImGui_ImplVulkan_Init(&initInfo);

    imguiInitialized = true;
}

void Renderer::drawFrame(const glm::mat4 &view, const glm::mat4 &proj,
                         const glm::vec3 &lightDir, const glm::vec3 &camPos)
{
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
                    UINT64_MAX);

    uint32_t imageIndex = 0;
    VkResult acquire = vkAcquireNextImageKHR(
        device, swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame],
        VK_NULL_HANDLE, &imageIndex);

    if (acquire == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapchain();
        return;
    }
    else if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR)
    {
        vkCheck(acquire, "vkAcquireNextImageKHR");
    }

    updateUniformBuffer(currentFrame, view, proj, lightDir, camPos);

    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    vkResetCommandBuffer(commandBuffers[currentFrame], 0);
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[imageIndex]};

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = waitSemaphores;
    submit.pWaitDstStageMask = waitStages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffers[currentFrame];
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = signalSemaphores;

    vkCheck(vkQueueSubmit(graphicsQueue, 1, &submit,
                          inFlightFences[currentFrame]),
            "vkQueueSubmit");

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = signalSemaphores;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain;
    present.pImageIndices = &imageIndex;

    VkResult presentResult = vkQueuePresentKHR(presentQueue, &present);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
        presentResult == VK_SUBOPTIMAL_KHR || framebufferResized)
    {
        framebufferResized = false;
        recreateSwapchain();
    }
    else
    {
        vkCheck(presentResult, "vkQueuePresentKHR");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// ---------------------------------------------------------------------------
// OpenXR per-eye rendering
// ---------------------------------------------------------------------------

void Renderer::createXrRenderPass()
{
    // Same 3-attachment MSAA layout as the desktop pass, but targeting the XR
    // swapchain color format and leaving the resolved image in
    // COLOR_ATTACHMENT_OPTIMAL (the layout the runtime expects on release).
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = xrColorFormat;
    colorAttachment.samples = msaaSamples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = depthFormat;
    depthAttachment.samples = msaaSamples;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription resolveAttachment{};
    resolveAttachment.format = xrColorFormat;
    resolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    resolveAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolveAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    resolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    resolveAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    resolveAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{
        1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkAttachmentReference resolveRef{2,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;
    subpass.pResolveAttachments = &resolveRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 3> attachments = {
        colorAttachment, depthAttachment, resolveAttachment};
    VkRenderPassCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = static_cast<uint32_t>(attachments.size());
    info.pAttachments = attachments.data();
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 1;
    info.pDependencies = &dependency;

    vkCheck(vkCreateRenderPass(device, &info, nullptr, &xrRenderPass),
            "vkCreateRenderPass (XR)");
}

void Renderer::createXrRenderTargets()
{
    for (uint32_t e = 0; e < kEyeCount; ++e)
    {
        const XrSystem::EyeSwapchain &sc = xr->eye(e);
        xrExtent[e] = {sc.width, sc.height};

        // Per-eye MSAA color + depth (separate per eye so the two eyes never
        // alias their shared targets).
        createImage(sc.width, sc.height, 1, xrColorFormat,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, xrMsaaColor[e].image,
                    xrMsaaColor[e].memory, msaaSamples);
        xrMsaaColor[e].view = createImageView(
            xrMsaaColor[e].image, xrColorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

        createImage(sc.width, sc.height, 1, depthFormat, VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, xrDepthImage[e],
                    xrDepthMemory[e], msaaSamples);
        xrDepthView[e] = createImageView(xrDepthImage[e], depthFormat,
                                         VK_IMAGE_ASPECT_DEPTH_BIT, 1);

        // One image view + framebuffer per runtime swapchain image.
        xrImageViews[e].resize(sc.images.size());
        xrFramebuffers[e].resize(sc.images.size());
        for (size_t i = 0; i < sc.images.size(); ++i)
        {
            xrImageViews[e][i] = createImageView(
                sc.images[i], xrColorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

            std::array<VkImageView, 3> attachments = {
                xrMsaaColor[e].view, xrDepthView[e], xrImageViews[e][i]};
            VkFramebufferCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            info.renderPass = xrRenderPass;
            info.attachmentCount = static_cast<uint32_t>(attachments.size());
            info.pAttachments = attachments.data();
            info.width = sc.width;
            info.height = sc.height;
            info.layers = 1;
            vkCheck(vkCreateFramebuffer(device, &info, nullptr,
                                        &xrFramebuffers[e][i]),
                    "vkCreateFramebuffer (XR)");
        }
    }
}

void Renderer::destroyXrRenderTargets()
{
    for (uint32_t e = 0; e < kEyeCount; ++e)
    {
        for (auto fb : xrFramebuffers[e])
            vkDestroyFramebuffer(device, fb, nullptr);
        xrFramebuffers[e].clear();
        for (auto v : xrImageViews[e])
            vkDestroyImageView(device, v, nullptr);
        xrImageViews[e].clear();

        destroyTexture(xrMsaaColor[e]);
        if (xrDepthView[e])
            vkDestroyImageView(device, xrDepthView[e], nullptr);
        if (xrDepthImage[e])
            vkDestroyImage(device, xrDepthImage[e], nullptr);
        if (xrDepthMemory[e])
            vkFreeMemory(device, xrDepthMemory[e], nullptr);
        xrDepthView[e] = VK_NULL_HANDLE;
        xrDepthImage[e] = VK_NULL_HANDLE;
        xrDepthMemory[e] = VK_NULL_HANDLE;
    }
}

void Renderer::createXrFrameResources()
{
    // Per-slot uniform buffers (frame-in-flight x eye), persistently mapped.
    VkDeviceSize size = sizeof(UniformBufferObject);
    xrUniformBuffers.resize(kXrSlots);
    xrUniformBuffersMemory.resize(kXrSlots);
    xrUniformBuffersMapped.resize(kXrSlots);
    for (uint32_t i = 0; i < kXrSlots; ++i)
    {
        createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     xrUniformBuffers[i], xrUniformBuffersMemory[i]);
        vkMapMemory(device, xrUniformBuffersMemory[i], 0, size, 0,
                    &xrUniformBuffersMapped[i]);
    }

    // Dedicated descriptor pool for the XR sets.
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = kXrSlots;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = kXrSlots * 4;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = kXrSlots;
    vkCheck(vkCreateDescriptorPool(device, &poolInfo, nullptr, &xrDescriptorPool),
            "vkCreateDescriptorPool (XR)");

    std::vector<VkDescriptorSetLayout> layouts(kXrSlots, descriptorSetLayout);
    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = xrDescriptorPool;
    alloc.descriptorSetCount = kXrSlots;
    alloc.pSetLayouts = layouts.data();
    xrDescriptorSets.resize(kXrSlots);
    vkCheck(vkAllocateDescriptorSets(device, &alloc, xrDescriptorSets.data()),
            "vkAllocateDescriptorSets (XR)");
    // Texture bindings are written later in createDescriptorSets() (uploadObject).

    // Command buffers: one per slot.
    xrCommandBuffers.resize(kXrSlots);
    VkCommandBufferAllocateInfo cbInfo{};
    cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool = commandPool;
    cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = kXrSlots;
    vkCheck(vkAllocateCommandBuffers(device, &cbInfo, xrCommandBuffers.data()),
            "vkAllocateCommandBuffers (XR)");

    // One fence per frame-in-flight (both eyes share it via a combined submit).
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vkCheck(vkCreateFence(device, &fenceInfo, nullptr, &xrInFlightFences[i]),
                "vkCreateFence (XR)");
    }
}

void Renderer::recordXrCommandBuffer(VkCommandBuffer cmd, uint32_t eye,
                                     uint32_t imageIndex, VkDescriptorSet set,
                                     uint32_t xrSlotIdx)
{
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCheck(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer (XR)");

    std::array<VkClearValue, 3> clears{};
    clears[0].color = {{0.05f, 0.05f, 0.08f, 1.0f}};
    clears[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = xrRenderPass;
    rpBegin.framebuffer = xrFramebuffers[eye][imageIndex];
    rpBegin.renderArea.extent = xrExtent[eye];
    rpBegin.clearValueCount = static_cast<uint32_t>(clears.size());
    rpBegin.pClearValues = clears.data();

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, xrPipeline);

    VkViewport viewport{};
    viewport.width = static_cast<float>(xrExtent[eye].width);
    viewport.height = static_cast<float>(xrExtent[eye].height);
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.extent = xrExtent[eye];
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VkBuffer vbs[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};

    uint32_t lastMeshId = UINT32_MAX;
    for (const DrawCall &dc : drawCalls)
    {
        const uint32_t mid = dc.meshId;
        if (mid != lastMeshId)
        {
            if (mid == 0)
            {
                vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, offsets);
                vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayout, 0, 1, &set, 0, nullptr);
            }
            else if (mid - 1 < extraSlots.size())
            {
                auto &es = extraSlots[mid - 1];
                vkCmdBindVertexBuffers(cmd, 0, 1, &es.vb, offsets);
                vkCmdBindIndexBuffer(cmd, es.ib, 0, VK_INDEX_TYPE_UINT32);
                // Extra slots have their own xrDescSets.
                const uint32_t safeIdx =
                    (xrSlotIdx < es.xrDescSets.size()) ? xrSlotIdx : 0u;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayout, 0, 1,
                                        &es.xrDescSets[safeIdx], 0, nullptr);
            }
            lastMeshId = mid;
        }
        const uint32_t ic =
            (mid == 0) ? indexCount
                       : (mid - 1 < extraSlots.size() ? extraSlots[mid - 1].indexCount : 0u);
        if (ic == 0)
            continue;
        PushConstants pc{dc.transform};
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, ic, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer (XR)");
}

void Renderer::renderXrFrame(XrSystem &xrSys, const PlayerRig &player,
                             const glm::vec3 &lightDir)
{
    XrSystem::FrameState fs = xrSys.beginFrame();
    if (!fs.shouldRender)
    {
        xrSys.endFrame(fs, false);
        return;
    }

    const uint32_t frame = xrCurrentFrame;
    vkWaitForFences(device, 1, &xrInFlightFences[frame], VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &xrInFlightFences[frame]);

    const glm::mat4 worldFromStage = player.worldFromStage();

    // Acquire + record both eyes, then submit together so a single fence covers
    // the whole frame.
    std::array<VkCommandBuffer, kEyeCount> cmds{};
    for (uint32_t eye = 0; eye < kEyeCount; ++eye)
    {
        const uint32_t img = xrSys.acquireImage(eye);
        const uint32_t slot = frame * kEyeCount + eye;

        const glm::mat4 worldFromView =
            worldFromStage * xrPoseToMatrix(fs.views[eye].pose);
        const glm::mat4 view = glm::inverse(worldFromView);
        const glm::mat4 proj =
            xrProjFromFov(fs.views[eye].fov, 0.01f, 1000.0f);
        const glm::vec3 eyePos = glm::vec3(worldFromView[3]);

        UniformBufferObject ubo{};
        ubo.view = view;
        ubo.proj = proj;
        ubo.lightDir = glm::vec4(lightDir, 0.0f);
        ubo.camPos = glm::vec4(eyePos, 1.0f);
        // Slot 0
        ubo.quantMin = glm::vec4(quantMin, 0.0f);
        ubo.quantExtent = glm::vec4(quantExtent, 0.0f);
        std::memcpy(xrUniformBuffersMapped[slot], &ubo, sizeof(ubo));
        // Extra slots
        for (auto &es : extraSlots)
        {
            if (slot < es.xrUbosMapped.size())
            {
                ubo.quantMin = glm::vec4(es.quantMin, 0.0f);
                ubo.quantExtent = glm::vec4(es.quantExtent, 0.0f);
                std::memcpy(es.xrUbosMapped[slot], &ubo, sizeof(ubo));
            }
        }

        cmds[eye] = xrCommandBuffers[slot];
        vkResetCommandBuffer(cmds[eye], 0);
        recordXrCommandBuffer(cmds[eye], eye, img, xrDescriptorSets[slot], slot);

        if (eye == 0)
        {
            lastEyeView = view;
            lastEyeProj = proj;
            lastEyePos = eyePos;
        }
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = kEyeCount;
    submit.pCommandBuffers = cmds.data();
    vkCheck(vkQueueSubmit(graphicsQueue, 1, &submit, xrInFlightFences[frame]),
            "vkQueueSubmit (XR)");

    for (uint32_t eye = 0; eye < kEyeCount; ++eye)
        xrSys.releaseImage(eye);

    xrSys.endFrame(fs, true);
    xrCurrentFrame = (frame + 1) % MAX_FRAMES_IN_FLIGHT;
}
