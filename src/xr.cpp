#include "include/xr.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <SDL3/SDL.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace
{

    void xrCheck(XrResult r, const char *what)
    {
        if (XR_FAILED(r))
        {
            throw std::runtime_error(std::string("OpenXR error (") +
                                     std::to_string(r) + ") in " + what);
        }
    }

    // Fetch an extension function pointer through the loader.
    template <typename Fn>
    Fn xrProc(XrInstance instance, const char *name)
    {
        PFN_xrVoidFunction fn = nullptr;
        if (XR_FAILED(xrGetInstanceProcAddr(instance, name, &fn)) || !fn)
        {
            throw std::runtime_error(std::string("xrGetInstanceProcAddr failed for ") +
                                     name);
        }
        return reinterpret_cast<Fn>(fn);
    }

    XrPath strToPath(XrInstance instance, const char *s)
    {
        XrPath p = XR_NULL_PATH;
        xrCheck(xrStringToPath(instance, s, &p), "xrStringToPath");
        return p;
    }

    constexpr XrViewConfigurationType kViewConfigType =
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;

} // namespace

XrSystem::~XrSystem()
{
    shutdownGraphics();
    if (xrInstance != XR_NULL_HANDLE)
        xrDestroyInstance(xrInstance);
}

void XrSystem::shutdownGraphics()
{
    destroyInput();
    // Destroy hand trackers before the session they were created from.
    if (pfnDestroyHandTracker)
    {
        for (auto &tracker : handTrackers)
        {
            if (tracker != XR_NULL_HANDLE)
            {
                pfnDestroyHandTracker(tracker);
                tracker = XR_NULL_HANDLE;
            }
        }
    }
    for (auto &sc : swapchains)
    {
        if (sc.handle != XR_NULL_HANDLE)
        {
            xrDestroySwapchain(sc.handle);
            sc.handle = XR_NULL_HANDLE;
        }
    }
    if (appSpace != XR_NULL_HANDLE)
    {
        xrDestroySpace(appSpace);
        appSpace = XR_NULL_HANDLE;
    }
    if (session != XR_NULL_HANDLE)
    {
        xrDestroySession(session);
        session = XR_NULL_HANDLE;
    }
}

bool XrSystem::create()
{
    // Check which optional extensions the runtime exposes before creating the
    // instance (so we can request them).  Enumeration may fail harmlessly when
    // no runtime is installed — we just skip optional extensions in that case.
    bool hasHandTracking = false;
    {
        uint32_t extCount = 0;
        if (XR_SUCCEEDED(xrEnumerateInstanceExtensionProperties(
                nullptr, 0, &extCount, nullptr)) &&
            extCount > 0)
        {
            std::vector<XrExtensionProperties> exts(
                extCount, {XR_TYPE_EXTENSION_PROPERTIES});
            xrEnumerateInstanceExtensionProperties(nullptr, extCount, &extCount,
                                                   exts.data());
            for (const auto &e : exts)
            {
                if (std::strcmp(e.extensionName,
                                XR_EXT_HAND_TRACKING_EXTENSION_NAME) == 0)
                {
                    hasHandTracking = true;
                }
            }
        }
    }

    // Build extension list: required + optional.
    std::vector<const char *> extensions = {XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME};
    if (hasHandTracking)
        extensions.push_back(XR_EXT_HAND_TRACKING_EXTENSION_NAME);

    XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
    ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ci.enabledExtensionNames = extensions.data();
    std::strncpy(ci.applicationInfo.applicationName, "ForceUnleashedVR",
                 XR_MAX_APPLICATION_NAME_SIZE - 1);
    ci.applicationInfo.applicationVersion = 1;
    ci.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;

    XrResult r = xrCreateInstance(&ci, &xrInstance);
    if (XR_FAILED(r))
    {
        SDL_Log("OpenXR: no runtime available (xrCreateInstance=%d); "
                "running desktop-only.",
                r);
        xrInstance = XR_NULL_HANDLE;
        return false;
    }

    // Load hand-tracking function pointers now that the instance exists.
    if (hasHandTracking)
    {
        auto load = [&](const char *name) -> PFN_xrVoidFunction
        {
            PFN_xrVoidFunction fn = nullptr;
            xrGetInstanceProcAddr(xrInstance, name, &fn);
            return fn;
        };
        pfnCreateHandTracker = reinterpret_cast<PFN_xrCreateHandTrackerEXT>(
            load("xrCreateHandTrackerEXT"));
        pfnDestroyHandTracker = reinterpret_cast<PFN_xrDestroyHandTrackerEXT>(
            load("xrDestroyHandTrackerEXT"));
        pfnLocateHandJoints = reinterpret_cast<PFN_xrLocateHandJointsEXT>(
            load("xrLocateHandJointsEXT"));
        if (pfnCreateHandTracker && pfnDestroyHandTracker && pfnLocateHandJoints)
        {
            handTrackingExtEnabled = true;
            SDL_Log("OpenXR: XR_EXT_hand_tracking extension enabled");
        }
    }

    XrSystemGetInfo sysInfo{XR_TYPE_SYSTEM_GET_INFO};
    sysInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    r = xrGetSystem(xrInstance, &sysInfo, &systemId);
    if (XR_FAILED(r))
    {
        SDL_Log("OpenXR: no HMD form factor available (xrGetSystem=%d); "
                "running desktop-only.",
                r);
        xrDestroyInstance(xrInstance);
        xrInstance = XR_NULL_HANDLE;
        systemId = XR_NULL_SYSTEM_ID;
        return false;
    }

    auto getReqs = xrProc<PFN_xrGetVulkanGraphicsRequirements2KHR>(
        xrInstance, "xrGetVulkanGraphicsRequirements2KHR");
    xrCheck(getReqs(xrInstance, systemId, &graphicsReqs),
            "xrGetVulkanGraphicsRequirements2KHR");

    XrSystemProperties props{XR_TYPE_SYSTEM_PROPERTIES};
    if (XR_SUCCEEDED(xrGetSystemProperties(xrInstance, systemId, &props)))
    {
        SDL_Log("OpenXR system: %s", props.systemName);
    }
    return true;
}

uint32_t XrSystem::requiredVulkanApiVersion() const
{
    // graphicsReqs uses XR_MAKE_VERSION (64-bit); map to a Vulkan apiVersion.
    const uint64_t v = graphicsReqs.minApiVersionSupported;
    const uint32_t major = static_cast<uint32_t>((v >> 48) & 0xffff);
    const uint32_t minor = static_cast<uint32_t>((v >> 32) & 0xffff);
    return VK_MAKE_API_VERSION(0, major, minor, 0);
}

VkInstance XrSystem::createVulkanInstance(const VkInstanceCreateInfo &ci)
{
    auto fn = xrProc<PFN_xrCreateVulkanInstanceKHR>(
        xrInstance, "xrCreateVulkanInstanceKHR");

    XrVulkanInstanceCreateInfoKHR info{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
    info.systemId = systemId;
    info.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
    info.vulkanCreateInfo = &ci;
    info.vulkanAllocator = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult vkResult = VK_SUCCESS;
    xrCheck(fn(xrInstance, &info, &instance, &vkResult),
            "xrCreateVulkanInstanceKHR");
    if (vkResult != VK_SUCCESS)
    {
        throw std::runtime_error("xrCreateVulkanInstanceKHR: inner vkCreateInstance failed");
    }
    return instance;
}

VkPhysicalDevice XrSystem::getVulkanPhysicalDevice(VkInstance instance)
{
    auto fn = xrProc<PFN_xrGetVulkanGraphicsDevice2KHR>(
        xrInstance, "xrGetVulkanGraphicsDevice2KHR");

    XrVulkanGraphicsDeviceGetInfoKHR info{
        XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
    info.systemId = systemId;
    info.vulkanInstance = instance;

    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    xrCheck(fn(xrInstance, &info, &gpu), "xrGetVulkanGraphicsDevice2KHR");
    return gpu;
}

VkDevice XrSystem::createVulkanDevice(VkPhysicalDevice gpu,
                                      const VkDeviceCreateInfo &ci)
{
    auto fn = xrProc<PFN_xrCreateVulkanDeviceKHR>(xrInstance,
                                                  "xrCreateVulkanDeviceKHR");

    XrVulkanDeviceCreateInfoKHR info{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
    info.systemId = systemId;
    info.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
    info.vulkanPhysicalDevice = gpu;
    info.vulkanCreateInfo = &ci;
    info.vulkanAllocator = nullptr;

    VkDevice device = VK_NULL_HANDLE;
    VkResult vkResult = VK_SUCCESS;
    xrCheck(fn(xrInstance, &info, &device, &vkResult),
            "xrCreateVulkanDeviceKHR");
    if (vkResult != VK_SUCCESS)
    {
        throw std::runtime_error("xrCreateVulkanDeviceKHR: inner vkCreateDevice failed");
    }
    return device;
}

void XrSystem::createSession(VkInstance instance, VkPhysicalDevice gpu,
                             VkDevice device, uint32_t queueFamilyIndex,
                             uint32_t queueIndex)
{
    XrGraphicsBindingVulkan2KHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
    binding.instance = instance;
    binding.physicalDevice = gpu;
    binding.device = device;
    binding.queueFamilyIndex = queueFamilyIndex;
    binding.queueIndex = queueIndex;

    XrSessionCreateInfo ci{XR_TYPE_SESSION_CREATE_INFO};
    ci.next = &binding;
    ci.systemId = systemId;
    xrCheck(xrCreateSession(xrInstance, &ci, &session), "xrCreateSession");

    // Check if the system supports hand tracking (requires the extension to be
    // enabled at instance creation time).
    if (handTrackingExtEnabled)
    {
        XrSystemHandTrackingPropertiesEXT handProps{
            XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT};
        XrSystemProperties sysProps{XR_TYPE_SYSTEM_PROPERTIES};
        sysProps.next = &handProps;
        if (XR_SUCCEEDED(
                xrGetSystemProperties(xrInstance, systemId, &sysProps)))
        {
            handTrackingSupported = handProps.supportsHandTracking == XR_TRUE;
        }
        SDL_Log("OpenXR hand tracking: %s",
                handTrackingSupported ? "supported" : "not supported");
        if (handTrackingSupported)
            createHandTrackers();
    }
}

void XrSystem::createReferenceSpace()
{
    // Prefer STAGE (room-scale, floor origin); fall back to LOCAL.
    uint32_t count = 0;
    xrEnumerateReferenceSpaces(session, 0, &count, nullptr);
    std::vector<XrReferenceSpaceType> spaces(count);
    xrEnumerateReferenceSpaces(session, count, &count, spaces.data());

    XrReferenceSpaceType chosen = XR_REFERENCE_SPACE_TYPE_LOCAL;
    for (auto s : spaces)
    {
        if (s == XR_REFERENCE_SPACE_TYPE_STAGE)
        {
            chosen = XR_REFERENCE_SPACE_TYPE_STAGE;
            break;
        }
    }

    XrReferenceSpaceCreateInfo ci{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    ci.referenceSpaceType = chosen;
    ci.poseInReferenceSpace.orientation.w = 1.0f; // identity
    xrCheck(xrCreateReferenceSpace(session, &ci, &appSpace),
            "xrCreateReferenceSpace");
    SDL_Log("OpenXR reference space: %s",
            chosen == XR_REFERENCE_SPACE_TYPE_STAGE ? "STAGE" : "LOCAL");
}

VkFormat XrSystem::createSwapchains(
    const std::vector<VkFormat> &preferredFormats)
{
    // View configuration views (recommended per-eye resolution).
    uint32_t viewCount = 0;
    xrCheck(xrEnumerateViewConfigurationViews(xrInstance, systemId,
                                              kViewConfigType, 0, &viewCount,
                                              nullptr),
            "xrEnumerateViewConfigurationViews(count)");
    if (viewCount != kEyeCount)
    {
        throw std::runtime_error("Expected 2 stereo views, got " +
                                 std::to_string(viewCount));
    }
    for (auto &v : viewConfigs)
        v.type = XR_TYPE_VIEW_CONFIGURATION_VIEW;
    xrCheck(xrEnumerateViewConfigurationViews(xrInstance, systemId,
                                              kViewConfigType, viewCount,
                                              &viewCount, viewConfigs.data()),
            "xrEnumerateViewConfigurationViews");

    // Pick a swapchain color format the runtime supports and we prefer.
    uint32_t fmtCount = 0;
    xrCheck(xrEnumerateSwapchainFormats(session, 0, &fmtCount, nullptr),
            "xrEnumerateSwapchainFormats(count)");
    std::vector<int64_t> runtimeFormats(fmtCount);
    xrCheck(xrEnumerateSwapchainFormats(session, fmtCount, &fmtCount,
                                        runtimeFormats.data()),
            "xrEnumerateSwapchainFormats");

    chosenFormat = VK_FORMAT_UNDEFINED;
    for (VkFormat pref : preferredFormats)
    {
        for (int64_t rf : runtimeFormats)
        {
            if (static_cast<VkFormat>(rf) == pref)
            {
                chosenFormat = pref;
                break;
            }
        }
        if (chosenFormat != VK_FORMAT_UNDEFINED)
            break;
    }
    if (chosenFormat == VK_FORMAT_UNDEFINED && !runtimeFormats.empty())
    {
        chosenFormat = static_cast<VkFormat>(runtimeFormats[0]);
    }

    for (uint32_t i = 0; i < kEyeCount; ++i)
    {
        EyeSwapchain &sc = swapchains[i];
        sc.width = viewConfigs[i].recommendedImageRectWidth;
        sc.height = viewConfigs[i].recommendedImageRectHeight;

        XrSwapchainCreateInfo ci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        ci.format = static_cast<int64_t>(chosenFormat);
        ci.sampleCount = 1; // we resolve our own MSAA into this
        ci.width = sc.width;
        ci.height = sc.height;
        ci.faceCount = 1;
        ci.arraySize = 1;
        ci.mipCount = 1;
        xrCheck(xrCreateSwapchain(session, &ci, &sc.handle),
                "xrCreateSwapchain");

        uint32_t imgCount = 0;
        xrCheck(xrEnumerateSwapchainImages(sc.handle, 0, &imgCount, nullptr),
                "xrEnumerateSwapchainImages(count)");
        std::vector<XrSwapchainImageVulkan2KHR> imgs(
            imgCount, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
        xrCheck(xrEnumerateSwapchainImages(
                    sc.handle, imgCount, &imgCount,
                    reinterpret_cast<XrSwapchainImageBaseHeader *>(imgs.data())),
                "xrEnumerateSwapchainImages");
        sc.images.clear();
        for (auto &img : imgs)
            sc.images.push_back(img.image);
    }

    SDL_Log("OpenXR swapchains: %ux%u per eye, format %d",
            swapchains[0].width, swapchains[0].height,
            static_cast<int>(chosenFormat));
    return chosenFormat;
}

void XrSystem::pollEvents(bool &exitRequested)
{
    XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
    auto next = [&]()
    {
        ev = {XR_TYPE_EVENT_DATA_BUFFER};
        return xrPollEvent(xrInstance, &ev) == XR_SUCCESS;
    };
    while (next())
    {
        switch (ev.type)
        {
        case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
            exitRequested = true;
            return;
        case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
        {
            auto &e = *reinterpret_cast<XrEventDataSessionStateChanged *>(&ev);
            state = e.state;
            switch (state)
            {
            case XR_SESSION_STATE_READY:
            {
                XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};
                bi.primaryViewConfigurationType = kViewConfigType;
                xrCheck(xrBeginSession(session, &bi), "xrBeginSession");
                running = true;
                break;
            }
            case XR_SESSION_STATE_STOPPING:
                running = false;
                xrCheck(xrEndSession(session), "xrEndSession");
                break;
            case XR_SESSION_STATE_EXITING:
            case XR_SESSION_STATE_LOSS_PENDING:
                exitRequested = true;
                break;
            default:
                break;
            }
            break;
        }
        default:
            break;
        }
    }
}

XrSystem::FrameState XrSystem::beginFrame()
{
    FrameState fs;

    XrFrameWaitInfo waitInfo{XR_TYPE_FRAME_WAIT_INFO};
    XrFrameState frameState{XR_TYPE_FRAME_STATE};
    xrCheck(xrWaitFrame(session, &waitInfo, &frameState), "xrWaitFrame");

    XrFrameBeginInfo beginInfo{XR_TYPE_FRAME_BEGIN_INFO};
    xrCheck(xrBeginFrame(session, &beginInfo), "xrBeginFrame");

    fs.displayTime = frameState.predictedDisplayTime;
    fs.shouldRender = frameState.shouldRender == XR_TRUE;
    lastDisplayTime = fs.displayTime;

    if (fs.shouldRender)
    {
        for (auto &v : fs.views)
            v.type = XR_TYPE_VIEW;
        XrViewLocateInfo locate{XR_TYPE_VIEW_LOCATE_INFO};
        locate.viewConfigurationType = kViewConfigType;
        locate.displayTime = fs.displayTime;
        locate.space = appSpace;
        XrViewState viewState{XR_TYPE_VIEW_STATE};
        uint32_t count = 0;
        xrCheck(xrLocateViews(session, &locate, &viewState, kEyeCount, &count,
                              fs.views.data()),
                "xrLocateViews");
        if (!(viewState.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) ||
            !(viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT))
        {
            fs.shouldRender = false; // pose not yet trackable
        }
    }
    locateHandJoints(fs.displayTime);
    return fs;
}

uint32_t XrSystem::acquireImage(uint32_t eye)
{
    XrSwapchainImageAcquireInfo acq{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    uint32_t index = 0;
    xrCheck(xrAcquireSwapchainImage(swapchains[eye].handle, &acq, &index),
            "xrAcquireSwapchainImage");
    XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wait.timeout = XR_INFINITE_DURATION;
    xrCheck(xrWaitSwapchainImage(swapchains[eye].handle, &wait),
            "xrWaitSwapchainImage");
    return index;
}

void XrSystem::releaseImage(uint32_t eye)
{
    XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrCheck(xrReleaseSwapchainImage(swapchains[eye].handle, &rel),
            "xrReleaseSwapchainImage");
}

void XrSystem::endFrame(const FrameState &fs, bool rendered)
{
    XrFrameEndInfo endInfo{XR_TYPE_FRAME_END_INFO};
    endInfo.displayTime = fs.displayTime;
    endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;

    const XrCompositionLayerBaseHeader *layers[1] = {nullptr};
    if (rendered)
    {
        for (uint32_t i = 0; i < kEyeCount; ++i)
        {
            projViews[i] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
            projViews[i].pose = fs.views[i].pose;
            projViews[i].fov = fs.views[i].fov;
            projViews[i].subImage.swapchain = swapchains[i].handle;
            projViews[i].subImage.imageRect.offset = {0, 0};
            projViews[i].subImage.imageRect.extent = {
                static_cast<int32_t>(swapchains[i].width),
                static_cast<int32_t>(swapchains[i].height)};
            projViews[i].subImage.imageArrayIndex = 0;
        }
        projLayer = {XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        projLayer.space = appSpace;
        projLayer.viewCount = kEyeCount;
        projLayer.views = projViews.data();
        layers[0] = reinterpret_cast<XrCompositionLayerBaseHeader *>(&projLayer);
        endInfo.layerCount = 1;
        endInfo.layers = layers;
    }
    else
    {
        endInfo.layerCount = 0;
        endInfo.layers = nullptr;
    }
    xrCheck(xrEndFrame(session, &endInfo), "xrEndFrame");
}

// ---------------------------------------------------------------------------
// Input (action sets)
// ---------------------------------------------------------------------------

void XrSystem::createInput()
{
    XrActionSetCreateInfo asci{XR_TYPE_ACTION_SET_CREATE_INFO};
    std::strncpy(asci.actionSetName, "gameplay", XR_MAX_ACTION_SET_NAME_SIZE - 1);
    std::strncpy(asci.localizedActionSetName, "Gameplay",
                 XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE - 1);
    asci.priority = 0;
    xrCheck(xrCreateActionSet(xrInstance, &asci, &actionSet),
            "xrCreateActionSet");

    handPaths[0] = strToPath(xrInstance, "/user/hand/left");
    handPaths[1] = strToPath(xrInstance, "/user/hand/right");

    auto makeAction = [&](const char *name, const char *localized,
                          XrActionType type, bool subactions)
    {
        XrActionCreateInfo aci{XR_TYPE_ACTION_CREATE_INFO};
        std::strncpy(aci.actionName, name, XR_MAX_ACTION_NAME_SIZE - 1);
        std::strncpy(aci.localizedActionName, localized,
                     XR_MAX_LOCALIZED_ACTION_NAME_SIZE - 1);
        aci.actionType = type;
        if (subactions)
        {
            aci.countSubactionPaths = static_cast<uint32_t>(handPaths.size());
            aci.subactionPaths = handPaths.data();
        }
        XrAction action = XR_NULL_HANDLE;
        xrCheck(xrCreateAction(actionSet, &aci, &action), "xrCreateAction");
        return action;
    };

    poseAction = makeAction("hand_pose", "Hand Pose",
                            XR_ACTION_TYPE_POSE_INPUT, true);
    moveAction = makeAction("move", "Move", XR_ACTION_TYPE_VECTOR2F_INPUT, false);
    turnAction = makeAction("turn", "Turn", XR_ACTION_TYPE_VECTOR2F_INPUT, false);
    selectAction =
        makeAction("select", "Select", XR_ACTION_TYPE_BOOLEAN_INPUT, false);
    // Per-hand grab trigger (both hands, controller fallback for hand tracking).
    grabAction =
        makeAction("grab", "Grab", XR_ACTION_TYPE_BOOLEAN_INPUT, true);

    struct Binding
    {
        XrAction action;
        const char *path;
    };
    // Suggested bindings for the common SteamVR-routed profiles.
    auto suggest = [&](const char *profile, const std::vector<Binding> &bs)
    {
        std::vector<XrActionSuggestedBinding> v;
        for (const auto &b : bs)
        {
            v.push_back({b.action, strToPath(xrInstance, b.path)});
        }
        XrInteractionProfileSuggestedBinding s{
            XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        s.interactionProfile = strToPath(xrInstance, profile);
        s.suggestedBindings = v.data();
        s.countSuggestedBindings = static_cast<uint32_t>(v.size());
        XrResult r = xrSuggestInteractionProfileBindings(xrInstance, &s);
        if (XR_FAILED(r))
        {
            SDL_Log("OpenXR: suggest bindings failed for %s (%d)", profile, r);
        }
    };

    // Valve Index: thumbstick + trackpad, trigger/squeeze.
    suggest("/interaction_profiles/valve/index_controller",
            {{poseAction, "/user/hand/left/input/grip/pose"},
             {poseAction, "/user/hand/right/input/grip/pose"},
             {moveAction, "/user/hand/left/input/thumbstick"},
             {turnAction, "/user/hand/right/input/thumbstick"},
             {selectAction, "/user/hand/right/input/trigger/click"},
             {selectAction, "/user/hand/right/input/trigger/value"},
             {grabAction, "/user/hand/left/input/trigger/click"},
             {grabAction, "/user/hand/right/input/trigger/click"},
             {grabAction, "/user/hand/left/input/trigger/value"},
             {grabAction, "/user/hand/right/input/trigger/value"},
             {grabAction, "/user/hand/left/input/squeeze/click"},
             {grabAction, "/user/hand/right/input/squeeze/click"},
             {grabAction, "/user/hand/left/input/squeeze/value"},
             {grabAction, "/user/hand/right/input/squeeze/value"}});
    // Oculus Touch.
    suggest("/interaction_profiles/oculus/touch_controller",
            {{poseAction, "/user/hand/left/input/grip/pose"},
             {poseAction, "/user/hand/right/input/grip/pose"},
             {moveAction, "/user/hand/left/input/thumbstick"},
             {turnAction, "/user/hand/right/input/thumbstick"},
             {selectAction, "/user/hand/right/input/trigger/click"},
             {selectAction, "/user/hand/right/input/trigger/value"},
             {grabAction, "/user/hand/left/input/squeeze/click"},
             {grabAction, "/user/hand/right/input/squeeze/click"},
             {grabAction, "/user/hand/left/input/squeeze/value"},
             {grabAction, "/user/hand/right/input/squeeze/value"}});
    // HTC Vive (trackpad, no thumbstick).
    suggest("/interaction_profiles/htc/vive_controller",
            {{poseAction, "/user/hand/left/input/grip/pose"},
             {poseAction, "/user/hand/right/input/grip/pose"},
             {moveAction, "/user/hand/left/input/trackpad"},
             {turnAction, "/user/hand/right/input/trackpad"},
             {selectAction, "/user/hand/right/input/trigger/click"},
             {selectAction, "/user/hand/right/input/trigger/value"},
             {grabAction, "/user/hand/left/input/trigger/click"},
             {grabAction, "/user/hand/right/input/trigger/click"},
             {grabAction, "/user/hand/left/input/trigger/value"},
             {grabAction, "/user/hand/right/input/trigger/value"}});
    // KHR simple controller (fallback; no thumbstick — pose + select only).
    suggest("/interaction_profiles/khr/simple_controller",
            {{poseAction, "/user/hand/left/input/grip/pose"},
             {poseAction, "/user/hand/right/input/grip/pose"},
             {selectAction, "/user/hand/right/input/select/click"},
             {grabAction, "/user/hand/left/input/select/click"},
             {grabAction, "/user/hand/right/input/select/click"}});

    // Pose action spaces (one per hand).
    for (uint32_t i = 0; i < 2; ++i)
    {
        XrActionSpaceCreateInfo sci{XR_TYPE_ACTION_SPACE_CREATE_INFO};
        sci.action = poseAction;
        sci.subactionPath = handPaths[i];
        sci.poseInActionSpace.orientation.w = 1.0f;
        xrCheck(xrCreateActionSpace(session, &sci, &handSpace[i]),
                "xrCreateActionSpace");
    }

    XrSessionActionSetsAttachInfo attach{
        XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
    attach.countActionSets = 1;
    attach.actionSets = &actionSet;
    xrCheck(xrAttachSessionActionSets(session, &attach),
            "xrAttachSessionActionSets");
}

void XrSystem::syncActions()
{
    move = glm::vec2(0.0f);
    turn = 0.0f;
    if (actionSet == XR_NULL_HANDLE || state != XR_SESSION_STATE_FOCUSED)
    {
        return;
    }

    XrActiveActionSet active{actionSet, XR_NULL_PATH};
    XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO};
    sync.countActiveActionSets = 1;
    sync.activeActionSets = &active;
    xrCheck(xrSyncActions(session, &sync), "xrSyncActions");

    XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
    gi.action = moveAction;
    XrActionStateVector2f mv{XR_TYPE_ACTION_STATE_VECTOR2F};
    if (XR_SUCCEEDED(xrGetActionStateVector2f(session, &gi, &mv)) &&
        mv.isActive)
    {
        const float dead = 0.2f;
        glm::vec2 v(mv.currentState.x, mv.currentState.y);
        if (glm::length(v) > dead)
            move = v;
    }

    gi.action = turnAction;
    XrActionStateVector2f tv{XR_TYPE_ACTION_STATE_VECTOR2F};
    if (XR_SUCCEEDED(xrGetActionStateVector2f(session, &gi, &tv)) &&
        tv.isActive)
    {
        const float snap = 0.6f;
        if (tv.currentState.x > snap && !turnLatched)
        {
            turn = 1.0f;
            turnLatched = true;
        }
        else if (tv.currentState.x < -snap && !turnLatched)
        {
            turn = -1.0f;
            turnLatched = true;
        }
        else if (tv.currentState.x > -snap && tv.currentState.x < snap)
        {
            turnLatched = false;
        }
    }
}

XrPosef XrSystem::handPose(Hand h, bool &valid) const
{
    XrPosef pose{};
    pose.orientation.w = 1.0f;
    valid = false;
    const uint32_t i = static_cast<uint32_t>(h);
    if (handSpace[i] == XR_NULL_HANDLE || appSpace == XR_NULL_HANDLE ||
        lastDisplayTime == 0)
    {
        return pose;
    }
    // Locate the hand action space in stage space at the latest predicted
    // display time.
    XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
    if (XR_FAILED(
            xrLocateSpace(handSpace[i], appSpace, lastDisplayTime, &loc)))
    {
        return pose;
    }
    const XrSpaceLocationFlags need = XR_SPACE_LOCATION_POSITION_VALID_BIT |
                                      XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
    if ((loc.locationFlags & need) == need)
    {
        pose = loc.pose;
        valid = true;
    }
    return pose;
}

void XrSystem::destroyInput()
{
    for (auto &s : handSpace)
    {
        if (s != XR_NULL_HANDLE)
        {
            xrDestroySpace(s);
            s = XR_NULL_HANDLE;
        }
    }
    if (actionSet != XR_NULL_HANDLE)
    {
        xrDestroyActionSet(actionSet);
        actionSet = XR_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------
// Hand tracking
// ---------------------------------------------------------------------------

void XrSystem::createHandTrackers()
{
    for (uint32_t i = 0; i < 2; ++i)
    {
        XrHandTrackerCreateInfoEXT ci{XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT};
        ci.hand = (i == 0) ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT;
        ci.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;
        XrResult r = pfnCreateHandTracker(session, &ci, &handTrackers[i]);
        if (XR_FAILED(r))
        {
            SDL_Log("OpenXR: xrCreateHandTrackerEXT failed (%d) for hand %u; "
                    "disabling hand tracking.",
                    r, i);
            handTrackingSupported = false;
            return;
        }
    }
    SDL_Log("OpenXR: hand trackers created");
}

void XrSystem::locateHandJoints(XrTime displayTime)
{
    if (!pfnLocateHandJoints || appSpace == XR_NULL_HANDLE)
        return;

    XrHandJointsLocateInfoEXT locateInfo{XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT};
    locateInfo.baseSpace = appSpace;
    locateInfo.time = displayTime;

    for (uint32_t i = 0; i < 2; ++i)
    {
        jointData[i].isTracked = false;
        if (handTrackers[i] == XR_NULL_HANDLE)
            continue;

        XrHandJointLocationsEXT locations{XR_TYPE_HAND_JOINT_LOCATIONS_EXT};
        locations.jointCount = XR_HAND_JOINT_COUNT_EXT;
        locations.jointLocations = jointData[i].joints.data();

        if (XR_SUCCEEDED(
                pfnLocateHandJoints(handTrackers[i], &locateInfo, &locations)))
        {
            jointData[i].isTracked = locations.isActive == XR_TRUE;
        }
    }
}

bool XrSystem::isGrabGesture(Hand h) const
{
    const uint32_t i = static_cast<uint32_t>(h);
    const auto &jd = jointData[i];

    // Hand-tracking path: fist gesture (INDEX_TIP and MIDDLE_TIP close to
    // PALM).  Only attempted when the runtime reports valid joint positions;
    // if hand tracking is active but no fist is detected we still fall through
    // to the controller check below so that holding a controller always works.
    if (jd.isTracked)
    {
        static constexpr float kFistThreshold = 0.065f;
        static constexpr XrSpaceLocationFlags kPosBit =
            XR_SPACE_LOCATION_POSITION_VALID_BIT;
        const auto &palm = jd.joints[XR_HAND_JOINT_PALM_EXT];
        const auto &idxTip = jd.joints[XR_HAND_JOINT_INDEX_TIP_EXT];
        const auto &midTip = jd.joints[XR_HAND_JOINT_MIDDLE_TIP_EXT];
        if ((palm.locationFlags & kPosBit) && (idxTip.locationFlags & kPosBit) &&
            (midTip.locationFlags & kPosBit))
        {
            auto dist3 = [](const XrVector3f &a, const XrVector3f &b)
            {
                const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
                return std::sqrt(dx * dx + dy * dy + dz * dz);
            };
            if (dist3(idxTip.pose.position, palm.pose.position) < kFistThreshold &&
                dist3(midTip.pose.position, palm.pose.position) < kFistThreshold)
            {
                return true;
            }
        }
        // Joints not valid or fist not detected — fall through to controller.
    }

    // Controller fallback (both hands): always checked so that pressing the
    // trigger works even when the runtime claims hand tracking is supported
    // (e.g. Meta Quest Link with controllers active).
    if (grabAction == XR_NULL_HANDLE || session == XR_NULL_HANDLE ||
        state != XR_SESSION_STATE_FOCUSED)
    {
        return false;
    }
    XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
    gi.action = grabAction;
    gi.subactionPath = handPaths[i];
    XrActionStateBoolean bs{XR_TYPE_ACTION_STATE_BOOLEAN};
    if (XR_SUCCEEDED(xrGetActionStateBoolean(session, &gi, &bs)) && bs.isActive)
    {
        return bs.currentState == XR_TRUE;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

glm::mat4 xrPoseToMatrix(const XrPosef &pose)
{
    glm::quat q(pose.orientation.w, pose.orientation.x, pose.orientation.y,
                pose.orientation.z);
    glm::vec3 p(pose.position.x, pose.position.y, pose.position.z);
    return glm::translate(glm::mat4(1.0f), p) * glm::mat4_cast(q);
}

glm::mat4 xrProjFromFov(const XrFovf &fov, float nearZ, float farZ)
{
    // Asymmetric frustum from the four FOV half-angles. Tangents because the
    // frustum is generally off-center. Produces Vulkan clip space: depth 0..1
    // (GLM_FORCE_DEPTH_ZERO_TO_ONE) with Y pointing down (no separate flip).
    const float l = std::tan(fov.angleLeft);
    const float r = std::tan(fov.angleRight);
    const float u = std::tan(fov.angleUp);
    const float d = std::tan(fov.angleDown);
    const float w = r - l;
    const float h = d - u; // note: down - up so +Y is down (Vulkan)

    glm::mat4 m(0.0f);
    m[0][0] = 2.0f / w;
    m[1][1] = 2.0f / h;
    m[2][0] = (r + l) / w;
    m[2][1] = (u + d) / h;
    m[2][2] = farZ / (nearZ - farZ);
    m[2][3] = -1.0f;
    m[3][2] = (farZ * nearZ) / (nearZ - farZ);
    return m;
}
