#ifndef XR_HPP
#define XR_HPP

#include <array>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <glm/glm.hpp>

// XrSystem owns every OpenXR object and the session state machine. It also acts
// as the authority for Vulkan instance/device/GPU selection: with
// XR_KHR_vulkan_enable2 the runtime must create those, so the Renderer routes
// its creation through the helpers here. When no runtime/HMD is present,
// create() returns false and the app falls back to a desktop-only path.
class XrSystem
{
public:
    static constexpr uint32_t kEyeCount = 2; // PRIMARY_STEREO

    enum class Hand
    {
        Left = 0,
        Right = 1,
        Count = 2
    };

    // Per-eye render target description handed to the Renderer so it can build
    // image views / framebuffers / MSAA + depth against the runtime's images.
    struct EyeSwapchain
    {
        XrSwapchain handle = XR_NULL_HANDLE;
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<VkImage> images; // runtime-owned color images
    };

    // Result of beginFrame(): whether to render and the per-eye view transforms.
    struct FrameState
    {
        bool shouldRender = false;
        XrTime displayTime = 0;
        std::array<XrView, kEyeCount> views{}; // pose + fov per eye (stage space)
    };

    // Per-hand joint data from XR_EXT_hand_tracking, updated every beginFrame.
    struct HandJointData
    {
        bool isTracked = false;
        std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
    };

    XrSystem() = default;
    ~XrSystem();

    XrSystem(const XrSystem &) = delete;
    XrSystem &operator=(const XrSystem &) = delete;

    // Phase 1: XrInstance + system + graphics requirements. Returns false if no
    // runtime is reachable or no HMD form factor is available (desktop
    // fallback). Throws only on genuinely unexpected runtime errors.
    bool create();

    bool available() const { return systemId != XR_NULL_SYSTEM_ID; }

    // --- Vulkan creation routed through OpenXR (called by the Renderer) ---
    // Each wraps the corresponding vkCreate* but lets the runtime inject the
    // instance/device extensions it requires. The caller fills `ci` with its own
    // layers/extensions (e.g. SDL surface ext + VK_KHR_swapchain for the mirror).
    VkInstance createVulkanInstance(const VkInstanceCreateInfo &ci);
    VkPhysicalDevice getVulkanPhysicalDevice(VkInstance instance);
    VkDevice createVulkanDevice(VkPhysicalDevice gpu,
                                const VkDeviceCreateInfo &ci);
    // Minimum Vulkan API version the runtime requires.
    uint32_t requiredVulkanApiVersion() const;

    // Phase 1: bind the session to the created Vulkan device.
    void createSession(VkInstance instance, VkPhysicalDevice gpu, VkDevice device,
                       uint32_t queueFamilyIndex, uint32_t queueIndex);

    // Phase 2: query view configuration and create per-eye swapchains. Picks a
    // color format from `supportedFormats` (the Renderer passes what its render
    // pass can target); returns the chosen VkFormat.
    VkFormat createSwapchains(const std::vector<VkFormat> &preferredFormats);
    void createReferenceSpace(); // STAGE, falling back to LOCAL

    const EyeSwapchain &eye(uint32_t i) const { return swapchains[i]; }
    VkFormat swapchainFormat() const { return chosenFormat; }

    // Phase 3: action set, suggested bindings, action spaces.
    void createInput();
    void syncActions();
    // Locomotion inputs since last sync (deadzoned thumbstick + edge-triggered
    // snap turn). Valid only while the session is focused.
    glm::vec2 moveInput() const { return move; }
    float turnInput() const { return turn; } // -1/0/+1 snap step this frame
    // Located hand pose in stage space; `valid` false if untracked.
    XrPosef handPose(Hand h, bool &valid) const;

    // Per-hand joint locations (XR_EXT_hand_tracking). isTracked is false when
    // hand tracking is unsupported or the hand is not visible.
    const HandJointData &getHandJoints(Hand h) const
    {
        return jointData[static_cast<uint32_t>(h)];
    }
    // True when the runtime reported XR_EXT_hand_tracking support.
    bool handTrackingAvailable() const { return handTrackingSupported; }
    // Fist-gesture (hand tracking) or trigger press (controller fallback).
    // Works for both hands regardless of whether hand tracking is active.
    bool isGrabGesture(Hand h) const;

    // --- per-frame orchestration ---
    // Pump the OpenXR event queue, advancing the session state machine. Sets
    // `exitRequested` when the runtime asks the app to quit.
    void pollEvents(bool &exitRequested);
    // True once the session is running (frames may be submitted).
    bool sessionRunning() const { return running; }
    // True when the session is focused (input should be read / acted upon).
    bool sessionFocused() const { return state == XR_SESSION_STATE_FOCUSED; }

    // xrWaitFrame + xrBeginFrame + xrLocateViews. Returns shouldRender=false when
    // the runtime wants the frame skipped (still must endFrame).
    FrameState beginFrame();
    // Acquire + wait the next image of eye `i`; returns its index into images[].
    uint32_t acquireImage(uint32_t eye);
    void releaseImage(uint32_t eye);
    // Submit the projection composition layer (or an empty frame when
    // shouldRender is false).
    void endFrame(const FrameState &fs, bool rendered);

    XrSpace stageSpace() const { return appSpace; }

    // Destroy every OpenXR object that references the Vulkan device (swapchains,
    // spaces, action set, session). Must be called before the Renderer destroys
    // the VkDevice. Idempotent; the destructor calls it too.
    void shutdownGraphics();

private:
    void destroyInput();
    void createHandTrackers();
    void locateHandJoints(XrTime displayTime);

    XrInstance xrInstance = XR_NULL_HANDLE;
    XrSystemId systemId = XR_NULL_SYSTEM_ID;
    XrSession session = XR_NULL_HANDLE;
    XrSpace appSpace = XR_NULL_HANDLE;
    XrSessionState state = XR_SESSION_STATE_UNKNOWN;
    bool running = false;

    XrGraphicsRequirementsVulkan2KHR graphicsReqs{
        XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR, nullptr, 0, 0};

    std::array<XrViewConfigurationView, kEyeCount> viewConfigs{};
    std::array<EyeSwapchain, kEyeCount> swapchains{};
    VkFormat chosenFormat = VK_FORMAT_UNDEFINED;

    // Composition-layer scratch, kept alive between beginFrame/endFrame.
    std::array<XrCompositionLayerProjectionView, kEyeCount> projViews{};
    XrCompositionLayerProjection projLayer{
        XR_TYPE_COMPOSITION_LAYER_PROJECTION, nullptr, 0, XR_NULL_HANDLE, 0,
        nullptr};

    // --- input ---
    XrActionSet actionSet = XR_NULL_HANDLE;
    XrAction poseAction = XR_NULL_HANDLE;
    XrAction moveAction = XR_NULL_HANDLE;
    XrAction turnAction = XR_NULL_HANDLE;
    XrAction selectAction = XR_NULL_HANDLE;
    XrAction grabAction = XR_NULL_HANDLE; // per-hand trigger (both hands)
    std::array<XrPath, 2> handPaths{};    // /user/hand/{left,right}
    std::array<XrSpace, 2> handSpace{XR_NULL_HANDLE, XR_NULL_HANDLE};

    glm::vec2 move{0.0f};
    float turn = 0.0f;
    bool turnLatched = false; // for snap-turn edge detection

    XrTime lastDisplayTime = 0; // updated each beginFrame, used to locate hands

    // --- hand tracking (XR_EXT_hand_tracking, optional) ---
    bool handTrackingExtEnabled = false;
    bool handTrackingSupported = false;
    PFN_xrCreateHandTrackerEXT pfnCreateHandTracker{};
    PFN_xrDestroyHandTrackerEXT pfnDestroyHandTracker{};
    PFN_xrLocateHandJointsEXT pfnLocateHandJoints{};
    std::array<XrHandTrackerEXT, 2> handTrackers{XR_NULL_HANDLE, XR_NULL_HANDLE};
    std::array<HandJointData, 2> jointData{};
};

// --- math helpers (defined in xr.cpp) ---
// XrPosef (orientation quat + position) -> world matrix.
glm::mat4 xrPoseToMatrix(const XrPosef &pose);
// Asymmetric projection from an XrFovf, Vulkan clip space (depth 0..1, Y down).
glm::mat4 xrProjFromFov(const XrFovf &fov, float nearZ, float farZ);

#endif
