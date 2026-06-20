#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include "include/model.hpp"
#include "include/object.hpp"
#include "include/player.hpp"
#include "include/renderer.hpp"
#include "include/xr.hpp"

namespace
{
    bool endsWith(const char *s, const char *suffix)
    {
        size_t ls = std::strlen(s), lf = std::strlen(suffix);
        return ls >= lf && std::strcmp(s + ls - lf, suffix) == 0;
    }

    // Place a model at an explicit world position with a pre-computed scale.
    glm::mat4 placeModelAt(const glm::vec3 &worldPos, float scale,
                           const glm::vec3 &center)
    {
        glm::mat4 m = glm::translate(glm::mat4(1.0f), worldPos);
        m = glm::scale(m, glm::vec3(scale));
        m = glm::translate(m, -center);
        return m;
    }

    // Desktop-fallback projection (no HMD): standard perspective, Vulkan clip.
    glm::mat4 desktopProj(float aspect)
    {
        glm::mat4 proj =
            glm::perspective(glm::radians(60.0f), aspect, 0.01f, 1000.0f);
        proj[1][1] *= -1.0f;
        return proj;
    }
} // namespace

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr,
                "Usage: %s <model.obj|.fbx|.glb|.gltf | object.toml>\n",
                argv[0]);
        return 1;
    }

    // A .toml describes a model plus the four PBR maps; any other path is a raw
    // model rendered with default (matte) textures.
    std::string modelPath = argv[1];
    TextureSet textures;
    if (endsWith(argv[1], ".toml"))
    {
        ObjectDef def;
        if (!loadObjectToml(argv[1], def))
        {
            return 1;
        }
        modelPath = def.modelPath;
        textures.base = def.base;
        textures.normal = def.normal;
        textures.roughness = def.roughness;
        textures.metallic = def.metallic;
    }

    Model model;
    if (!loadModel(modelPath.c_str(), model))
    {
        return 1;
    }

    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
        return 1;
    }

    SDL_Window *window =
        SDL_CreateWindow("ForceUnleashedVR", 1600, 900,
                         SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window)
    {
        SDL_Log("Could not create window: %s", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // OpenXR: create() returns false when no runtime/HMD is reachable, in which
    // case the app runs desktop-only with a free-look FPS camera.
    XrSystem xr;
    const bool vrAvailable = xr.create();

    // The player rig: spawns at the stage origin facing -Z (toward the model).
    PlayerRig player;

    // Light travelling downward (from above) with a slight angle so the shading
    // reads as 3D rather than flat-on-top.
    const glm::vec3 lightDir = glm::normalize(glm::vec3(-0.3f, -1.0f, -0.3f));

    try
    {
        Renderer renderer(window, vrAvailable ? &xr : nullptr);
        renderer.uploadObject(model, textures);

        // Drop the model ~1.5 m in front of spawn at roughly eye height, scaled
        // so its bounding sphere is ~0.5 m across.
        static constexpr float kObjectDesiredRadius = 0.5f;
        static constexpr float kGrabRange = 0.35f;
        const float objectScale =
            model.radius > 0.0001f ? kObjectDesiredRadius / model.radius : 1.0f;
        glm::vec3 objectWorldPos{0.0f, 1.2f, -1.5f};
        bool objectGrabbed = false;
        auto grabbingHand = XrSystem::Hand::Right;
        glm::vec3 grabHandOffset{0.0f};

        renderer.setObjectTransform(
            placeModelAt(objectWorldPos, objectScale, model.center));

        bool quit = false;
        bool dragging = false;
        SDL_Event e;

        // Build the list of MSAA options the hardware supports.
        const VkSampleCountFlagBits hwMax = renderer.getMaxMsaaSamples();
        struct MsaaOption
        {
            const char *label;
            VkSampleCountFlagBits count;
        };
        const MsaaOption msaaOptions[] = {
            {"Off (1x)", VK_SAMPLE_COUNT_1_BIT},
            {"2x MSAA", VK_SAMPLE_COUNT_2_BIT},
            {"4x MSAA", VK_SAMPLE_COUNT_4_BIT},
            {"8x MSAA", VK_SAMPLE_COUNT_8_BIT},
        };
        int msaaOptionCount = 1;
        for (const auto &opt : msaaOptions)
            if (opt.count <= hwMax)
                msaaOptionCount++;
        msaaOptionCount = std::min(msaaOptionCount,
                                   static_cast<int>(std::size(msaaOptions)));

        auto sampleToIndex = [&](VkSampleCountFlagBits c)
        {
            for (int i = 0; i < msaaOptionCount; ++i)
                if (msaaOptions[i].count == c)
                    return i;
            return 0;
        };

        int msaaSelected = sampleToIndex(renderer.getMsaaSamples());
        VkSampleCountFlagBits pendingMsaa = VK_SAMPLE_COUNT_FLAG_BITS_MAX_ENUM;

        // Mesh LOD state.
        float lodRatio = 1.0f;
        float lodError = 0.02f;
        bool pendingLod = false;
        Model lodModel;

        uint64_t prevTicks = SDL_GetTicks();

        while (!quit)
        {
            if (pendingMsaa != VK_SAMPLE_COUNT_FLAG_BITS_MAX_ENUM)
            {
                renderer.setMsaaSamples(pendingMsaa);
                msaaSelected = sampleToIndex(renderer.getMsaaSamples());
                pendingMsaa = VK_SAMPLE_COUNT_FLAG_BITS_MAX_ENUM;
            }

            if (pendingLod)
            {
                if (lodRatio >= 0.999f)
                {
                    renderer.updateMesh(model);
                }
                else
                {
                    simplifyModel(model, lodRatio, lodError, lodModel);
                    renderer.updateMesh(lodModel);
                }
                pendingLod = false;
            }

            while (SDL_PollEvent(&e))
            {
                ImGui_ImplSDL3_ProcessEvent(&e);

                switch (e.type)
                {
                case SDL_EVENT_QUIT:
                    quit = true;
                    break;
                case SDL_EVENT_KEY_DOWN:
                    if (e.key.key == SDLK_ESCAPE)
                        quit = true;
                    else if (e.key.key == SDLK_Q)
                        player.snapTurn(-1.0f); // turn left
                    else if (e.key.key == SDLK_E)
                        player.snapTurn(1.0f); // turn right
                    break;
                case SDL_EVENT_WINDOW_RESIZED:
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                    renderer.onResize();
                    break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                    if (e.button.button == SDL_BUTTON_LEFT &&
                        !ImGui::GetIO().WantCaptureMouse)
                        dragging = true;
                    break;
                case SDL_EVENT_MOUSE_BUTTON_UP:
                    if (e.button.button == SDL_BUTTON_LEFT)
                        dragging = false;
                    break;
                case SDL_EVENT_MOUSE_MOTION:
                    if (dragging)
                    {
                        // Desktop mouse-look (no effect on HMD orientation).
                        player.addYaw(e.motion.xrel * 0.0035f);
                        player.addPitch(-e.motion.yrel * 0.0035f);
                    }
                    break;
                default:
                    break;
                }
            }

            // Advance the OpenXR session state machine.
            if (vrAvailable)
            {
                bool exitRequested = false;
                xr.pollEvents(exitRequested);
                if (exitRequested)
                    quit = true;
            }

            const uint64_t now = SDL_GetTicks();
            const float dt = (now - prevTicks) / 1000.0f;
            prevTicks = now;

            // Keyboard locomotion (WASD), shared by VR and desktop.
            const bool *ks = SDL_GetKeyboardState(nullptr);
            glm::vec2 kbMove(0.0f);
            if (ks[SDL_SCANCODE_W])
                kbMove.y += 1.0f;
            if (ks[SDL_SCANCODE_S])
                kbMove.y -= 1.0f;
            if (ks[SDL_SCANCODE_D])
                kbMove.x += 1.0f;
            if (ks[SDL_SCANCODE_A])
                kbMove.x -= 1.0f;
            if (!ImGui::GetIO().WantCaptureKeyboard)
                player.moveLocal(kbMove, dt);

            // VR frame: controller locomotion + stereo render.
            const bool xrRunning = vrAvailable && xr.sessionRunning();
            if (xrRunning)
            {
                xr.syncActions();
                player.moveLocal(xr.moveInput(), dt);
                player.snapTurn(xr.turnInput());
                renderer.renderXrFrame(xr, player, lightDir);

                // --- grab mechanic ---
                // Returns world-space palm position; falls back to grip pose.
                auto getHandPos = [&](XrSystem::Hand h, glm::vec3 &pos) -> bool
                {
                    const auto &jd = xr.getHandJoints(h);
                    if (jd.isTracked)
                    {
                        const auto &palm = jd.joints[XR_HAND_JOINT_PALM_EXT];
                        if (palm.locationFlags &
                            XR_SPACE_LOCATION_POSITION_VALID_BIT)
                        {
                            pos = {palm.pose.position.x, palm.pose.position.y,
                                   palm.pose.position.z};
                            return true;
                        }
                    }
                    bool valid = false;
                    const XrPosef grip = xr.handPose(h, valid);
                    if (valid)
                    {
                        pos = {grip.position.x, grip.position.y,
                               grip.position.z};
                        return true;
                    }
                    return false;
                };

                if (!objectGrabbed)
                {
                    using Hand = XrSystem::Hand;
                    for (int hi = 0; hi < static_cast<int>(Hand::Count); ++hi)
                    {
                        const auto h = static_cast<Hand>(hi);
                        glm::vec3 hpos{0.0f};
                        if (!getHandPos(h, hpos))
                            continue;
                        if (glm::distance(hpos, objectWorldPos) < kGrabRange &&
                            xr.isGrabGesture(h))
                        {
                            objectGrabbed = true;
                            grabbingHand = h;
                            grabHandOffset = objectWorldPos - hpos;
                            break;
                        }
                    }
                }
                else
                {
                    glm::vec3 hpos{0.0f};
                    if (!getHandPos(grabbingHand, hpos) ||
                        !xr.isGrabGesture(grabbingHand))
                    {
                        objectGrabbed = false;
                    }
                    else
                    {
                        objectWorldPos = hpos + grabHandOffset;
                        renderer.setObjectTransform(
                            placeModelAt(objectWorldPos, objectScale,
                                         model.center));
                    }
                }
            }

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplSDL3_NewFrame();
            ImGui::NewFrame();

            ImGui::Begin("Settings");
            ImGui::Text("Mode: %s", xrRunning ? "VR (OpenXR)"
                                              : (vrAvailable ? "VR (waiting)"
                                                             : "Desktop"));

            ImGui::SeparatorText("Rendering");
            const char *msaaLabels[std::size(msaaOptions) + 1]{};
            for (int i = 0; i < msaaOptionCount; ++i)
                msaaLabels[i] = msaaOptions[i].label;
            if (ImGui::Combo("Anti-Aliasing", &msaaSelected, msaaLabels,
                             msaaOptionCount))
                pendingMsaa = msaaOptions[msaaSelected].count;

            ImGui::SeparatorText("Mesh / LOD");
            ImGui::SliderFloat("Simplify", &lodRatio, 0.05f, 1.0f, "%.2f");
            ImGui::SliderFloat("Error", &lodError, 0.0f, 0.20f, "%.2f");
            if (ImGui::Button("Apply LOD"))
                pendingLod = true;
            ImGui::Text("Tris: %zu -> %u", model.indices.size() / 3,
                        renderer.getIndexCount() / 3);

            ImGui::SeparatorText("Player");
            ImGui::Text("Pos: %.2f %.2f %.2f", player.position.x,
                        player.position.y, player.position.z);

            if (xrRunning)
            {
                ImGui::SeparatorText("Grab");
                ImGui::Text("Hand tracking: %s",
                            xr.handTrackingAvailable() ? "yes" : "no");
                ImGui::Text("Grabbed: %s",
                            objectGrabbed
                                ? (grabbingHand == XrSystem::Hand::Left
                                       ? "left hand"
                                       : "right hand")
                                : "no");
            }

            ImGui::End();
            ImGui::Render();

            // Desktop window: mirror the left eye while in VR, otherwise a
            // free-look FPS view.
            glm::mat4 view, proj;
            glm::vec3 camPos;
            if (xrRunning)
            {
                view = renderer.lastEyeView;
                proj = renderer.lastEyeProj;
                camPos = renderer.lastEyePos;
            }
            else
            {
                int w = 0, h = 0;
                SDL_GetWindowSizeInPixels(window, &w, &h);
                float aspect =
                    h > 0 ? static_cast<float>(w) / static_cast<float>(h) : 1.0f;
                view = player.eyeView();
                proj = desktopProj(aspect);
                camPos = player.eyePosition();
            }
            renderer.drawFrame(view, proj, lightDir, camPos);
        }
    }
    catch (const std::exception &ex)
    {
        SDL_Log("Fatal: %s", ex.what());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
