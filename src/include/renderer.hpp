#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include "model.hpp"

struct SDL_Window;
class XrSystem;
class PlayerRig;

// The four metallic-roughness PBR maps for an object. Paths are loaded from
// disk; if a path is empty the renderer substitutes a default constant texture.
struct TextureSet
{
    std::string base;
    std::string normal;
    std::string roughness;
    std::string metallic;
};

// Associates a mesh slot with a per-draw world transform.
struct DrawCall
{
    uint32_t meshId = 0;
    glm::mat4 transform{1.0f};
};

// Minimal Vulkan renderer for an indexed mesh shaded with metallic-roughness
// PBR under one directional light. The mesh can be drawn multiple times with
// per-object transforms. Owns all Vulkan objects and tears them down in the
// destructor.
class Renderer
{
public:
    // `xr` may be null (or an unavailable system) for desktop-only operation;
    // when present, the Vulkan instance/device are created through OpenXR and an
    // XR per-eye render path is set up.
    explicit Renderer(SDL_Window *window, XrSystem *xr = nullptr);
    ~Renderer();

    Renderer(const Renderer &) = delete;
    Renderer &operator=(const Renderer &) = delete;

    // Upload the mesh + textures and wire up descriptors. Call once before the
    // first drawFrame. The position-dequantization bbox is computed from this
    // (full-resolution) model and reused for any later updateMesh() calls.
    void uploadObject(const Model &model, const TextureSet &textures);

    // Replace just the vertex/index buffers (e.g. after LOD simplification).
    // Textures, descriptors and the quantization bbox are left unchanged.
    void updateMesh(const Model &model);

    // Number of indices currently uploaded (for UI readouts).
    uint32_t getIndexCount() const { return indexCount; }

    // Render one frame with the given camera matrices, light direction, and
    // camera world position (desktop window / mirror).
    void drawFrame(const glm::mat4 &view, const glm::mat4 &proj,
                   const glm::vec3 &lightDir, const glm::vec3 &camPos);

    // Render the stereo HMD frame through OpenXR (wait/begin/locate/2 eyes/end).
    // After it returns, `lastEyeView`/`lastEyeProj`/`lastEyePos` hold the left
    // eye's matrices so the caller can mirror them to the desktop window.
    void renderXrFrame(XrSystem &xr, const PlayerRig &player,
                       const glm::vec3 &lightDir);
    glm::mat4 lastEyeView{1.0f};
    glm::mat4 lastEyeProj{1.0f};
    glm::vec3 lastEyePos{0.0f};

    // Upload an additional mesh and return its meshId (1, 2, …).
    // uploadObject must be called first to initialise slot 0.
    uint32_t uploadMesh(const Model &model, const TextureSet &textures);

    // Set the list of draw calls used by the next frame.
    void setDrawCalls(const std::vector<DrawCall> &calls)
    {
        drawCalls = calls.empty() ? std::vector<DrawCall>{{0, glm::mat4(1.0f)}} : calls;
    }

    // Backward-compatible helpers — all draws use mesh slot 0.
    void setObjectTransform(const glm::mat4 &m) { drawCalls = {{0, m}}; }
    void setObjectTransforms(const std::vector<glm::mat4> &transforms)
    {
        drawCalls.clear();
        for (const auto &t : transforms)
            drawCalls.push_back({0, t});
        if (drawCalls.empty())
            drawCalls = {{0, glm::mat4(1.0f)}};
    }

    bool hasXr() const { return xr != nullptr; }

    // Mark the swapchain as needing recreation (window resized).
    void onResize() { framebufferResized = true; }

    VkSampleCountFlagBits getMsaaSamples() const { return msaaSamples; }
    VkSampleCountFlagBits getMaxMsaaSamples() const;
    void setMsaaSamples(VkSampleCountFlagBits samples);

private:
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    struct UniformBufferObject
    {
        glm::mat4 view;
        glm::mat4 proj;
        glm::vec4 lightDir;
        glm::vec4 camPos;
        glm::vec4 quantMin;    // xyz: bbox min for position dequantization
        glm::vec4 quantExtent; // xyz: bbox extent (max - min)
    };

    struct Texture
    {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        uint32_t mipLevels = 1;
    };

    // --- setup steps (called from the constructor) ---
    void createInstance();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void createMsaaColorResources();
    void createDepthResources();
    void createRenderPass();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createUniformBuffers();
    void createTextureSampler();
    void createDescriptorPool();
    void createCommandBuffers();
    void createSyncObjects();
    void initImGui();

    void recreateSwapchain();
    void cleanupSwapchain();
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void updateUniformBuffer(uint32_t frame, const glm::mat4 &view,
                             const glm::mat4 &proj, const glm::vec3 &lightDir,
                             const glm::vec3 &camPos);

    // Build a graphics pipeline (reusing pipelineLayout) for a given render pass
    // + sample count. Shared by the desktop and XR paths.
    VkPipeline buildPipeline(VkRenderPass rp, VkSampleCountFlagBits samples);

    // --- XR setup / teardown (only when xr != nullptr) ---
    void createXrRenderPass();
    void createXrRenderTargets(); // per-eye MSAA + depth + image views + FBs
    void destroyXrRenderTargets();
    void createXrFrameResources(); // UBOs, descriptor sets, cmd buffers, fences
    void recordXrCommandBuffer(VkCommandBuffer cmd, uint32_t eye,
                               uint32_t imageIndex, VkDescriptorSet set,
                               uint32_t xrSlotIdx);

    // Quantize `model` into PackedVertex form and (re)create the vertex/index
    // buffers, destroying any existing ones first. Sets indexCount.
    void createMeshBuffers(const Model &model);

    // --- buffer / command helpers ---
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, VkBuffer &buffer,
                      VkDeviceMemory &memory);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cmd);
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
    VkShaderModule createShaderModule(const char *spvPath);

    // --- image / texture helpers ---
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
                     VkFormat format, VkImageTiling tiling,
                     VkImageUsageFlags usage, VkMemoryPropertyFlags props,
                     VkImage &image, VkDeviceMemory &memory,
                     VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);
    VkImageView createImageView(VkImage image, VkFormat format,
                                VkImageAspectFlags aspect, uint32_t mipLevels);
    void transitionImageLayout(VkImage image, VkImageLayout oldLayout,
                               VkImageLayout newLayout, uint32_t mipLevels);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width,
                           uint32_t height);
    void generateMipmaps(VkImage image, VkFormat format, int32_t width,
                         int32_t height, uint32_t mipLevels);
    // Loads `path` if non-empty; otherwise builds a 1x1 texture of `fallback`.
    Texture loadOrDefaultTexture(const std::string &path, VkFormat format,
                                 const uint8_t fallback[4]);
    Texture createSolidTexture(const uint8_t rgba[4], VkFormat format);
    void destroyTexture(Texture &tex);
    void createDescriptorSets();

    // Extra mesh slot (meshId >= 1): geometry + textures + own descriptor sets.
    // Slot 0 uses the existing vertex/index/texture/descriptor members below.
    struct ExtraSlot
    {
        VkBuffer vb = VK_NULL_HANDLE, ib = VK_NULL_HANDLE;
        VkDeviceMemory vbMem = VK_NULL_HANDLE, ibMem = VK_NULL_HANDLE;
        uint32_t indexCount = 0;
        glm::vec3 quantMin{0.0f}, quantExtent{1.0f};
        Texture pbr[4];
        VkDescriptorPool descPool = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> descSets;   // MAX_FRAMES_IN_FLIGHT
        std::vector<VkDescriptorSet> xrDescSets; // kXrSlots
        std::vector<VkBuffer> ubos;              // MAX_FRAMES_IN_FLIGHT
        std::vector<VkDeviceMemory> ubosMem;
        std::vector<void *> ubosMapped;
        std::vector<VkBuffer> xrUbos; // kXrSlots
        std::vector<VkDeviceMemory> xrUbosMem;
        std::vector<void *> xrUbosMapped;
    };

    SDL_Window *window = nullptr;
    XrSystem *xr = nullptr;

    // Active draw calls for the next frame.
    std::vector<DrawCall> drawCalls{{0, glm::mat4(1.0f)}};
    // Additional mesh slots (meshId 1+).
    std::vector<ExtraSlot> extraSlots;

    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    uint32_t graphicsFamily = 0;
    uint32_t presentFamily = 0;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent{};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    std::vector<VkFramebuffer> framebuffers;

    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;
    uint32_t indexCount = 0;

    // Position-dequantization bbox, computed once from the full-res model.
    glm::vec3 quantMin{0.0f};
    glm::vec3 quantExtent{1.0f};

    // PBR textures: [0]=base, [1]=normal, [2]=roughness, [3]=metallic.
    Texture pbrTextures[4];
    VkSampler textureSampler = VK_NULL_HANDLE;

    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    Texture msaaColor;

    bool imguiInitialized = false;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void *> uniformBuffersMapped;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    uint32_t currentFrame = 0;
    bool framebufferResized = false;
    bool validationEnabled = false;

    // --- XR per-eye rendering (only populated when xr != nullptr) ---
    static constexpr uint32_t kEyeCount = 2;

    VkFormat xrColorFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D xrExtent[kEyeCount]{};
    VkRenderPass xrRenderPass = VK_NULL_HANDLE;
    VkPipeline xrPipeline = VK_NULL_HANDLE;

    // Per-eye render targets.
    Texture xrMsaaColor[kEyeCount];
    VkImage xrDepthImage[kEyeCount]{};
    VkDeviceMemory xrDepthMemory[kEyeCount]{};
    VkImageView xrDepthView[kEyeCount]{};
    std::vector<VkImageView> xrImageViews[kEyeCount];
    std::vector<VkFramebuffer> xrFramebuffers[kEyeCount];

    // Per (frame-in-flight × eye) frame resources.
    static constexpr uint32_t kXrSlots = MAX_FRAMES_IN_FLIGHT * kEyeCount;
    std::vector<VkBuffer> xrUniformBuffers;
    std::vector<VkDeviceMemory> xrUniformBuffersMemory;
    std::vector<void *> xrUniformBuffersMapped;
    std::vector<VkDescriptorSet> xrDescriptorSets;
    std::vector<VkCommandBuffer> xrCommandBuffers;
    VkFence xrInFlightFences[MAX_FRAMES_IN_FLIGHT]{};
    VkDescriptorPool xrDescriptorPool = VK_NULL_HANDLE;
    uint32_t xrCurrentFrame = 0;
};

#endif
