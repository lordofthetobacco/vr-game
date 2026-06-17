#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include "model.hpp"

struct SDL_Window;

// The four metallic-roughness PBR maps for an object. Paths are loaded from
// disk; if a path is empty the renderer substitutes a default constant texture.
struct TextureSet {
    std::string base;
    std::string normal;
    std::string roughness;
    std::string metallic;
};

// Minimal Vulkan renderer for a single indexed mesh shaded with metallic-
// roughness PBR under one directional light. Owns all Vulkan objects and tears
// them down in the destructor.
class Renderer {
public:
    explicit Renderer(SDL_Window *window);
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
    // camera world position.
    void drawFrame(const glm::mat4 &view, const glm::mat4 &proj,
                   const glm::vec3 &lightDir, const glm::vec3 &camPos);

    // Mark the swapchain as needing recreation (window resized).
    void onResize() { framebufferResized = true; }

    VkSampleCountFlagBits getMsaaSamples() const { return msaaSamples; }
    VkSampleCountFlagBits getMaxMsaaSamples() const;
    void setMsaaSamples(VkSampleCountFlagBits samples);

private:
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    struct UniformBufferObject {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
        glm::vec4 lightDir;
        glm::vec4 camPos;
        glm::vec4 quantMin;    // xyz: bbox min for position dequantization
        glm::vec4 quantExtent; // xyz: bbox extent (max - min)
    };

    struct Texture {
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

    SDL_Window *window = nullptr;

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
};

#endif
