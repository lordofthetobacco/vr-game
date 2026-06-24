#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include "include/model.hpp"
#include "include/object.hpp"
#include "include/physics_world.hpp"
#include "include/player.hpp"
#include "include/renderer.hpp"
#include "include/scene.hpp"
#include "include/xr.hpp"

// Bullet3 needed only for CF_KINEMATIC_OBJECT flag manipulation on grab/release.
#include <btBulletDynamicsCommon.h>

namespace
{
    bool endsWith(const char *s, const char *suffix)
    {
        size_t ls = std::strlen(s), lf = std::strlen(suffix);
        return ls >= lf && std::strcmp(s + ls - lf, suffix) == 0;
    }

    // Detect whether a .toml file is a scene file (has [[objects]]) vs a
    // single-object file.
    bool isSceneToml(const char *path)
    {
        std::FILE *f = std::fopen(path, "r");
        if (!f)
            return false;
        char buf[256];
        while (std::fgets(buf, sizeof(buf), f))
        {
            // Detect [objects.name] dictionary format or old [[objects]] array.
            if (std::strstr(buf, "[objects.") || std::strstr(buf, "[[objects]]"))
            {
                std::fclose(f);
                return true;
            }
        }
        std::fclose(f);
        return false;
    }

    // Convert a scene object's transform component to a renderable model matrix.
    glm::mat4 entityMatrix(const Entity &e)
    {
        glm::mat4 m = transformToMatrix(e.transform);
        m = glm::translate(m, -e.center);
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

    // Spawn an object a fixed distance in front of the current eye pose.
    glm::vec3 placeInFrontOfEye(const glm::vec3 &eyePos, const glm::mat4 &view,
                                float distance, float verticalOffset)
    {
        const glm::mat4 worldFromEye = glm::inverse(view);
        glm::vec3 forward = -glm::vec3(worldFromEye[2]);
        forward.y = 0.0f;
        const float forwardLen = glm::length(forward);
        if (forwardLen < 0.0001f)
            forward = glm::vec3(0.0f, 0.0f, -1.0f);
        else
            forward /= forwardLen;
        return eyePos + forward * distance +
               glm::vec3(0.0f, verticalOffset, 0.0f);
    }
} // namespace

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr,
                "Usage: %s <model.obj|.fbx|.glb|.gltf | object.toml | scene.toml>\n",
                argv[0]);
        return 1;
    }

    // --- Load scene definition -----------------------------------------------
    // If it's a scene TOML (has [[objects]]) use the new loader;
    // otherwise fall back to the single-object path (backward compat).
    SceneDef sceneDef;
    if (endsWith(argv[1], ".toml") && isSceneToml(argv[1]))
    {
        if (!loadSceneToml(argv[1], sceneDef))
            return 1;
    }
    else
    {
        // Legacy path: single-object file or bare model.
        std::string modelPath = argv[1];
        ObjectDef objectDef;
        if (endsWith(argv[1], ".toml"))
        {
            if (!loadObjectToml(argv[1], objectDef))
                return 1;
            modelPath = objectDef.modelPath;
        }
        // Build HandsDef from ObjectDef fields (backward compat).
        HandsDef hands;
        hands.enabled = objectDef.handsEnabled;
        hands.radius = objectDef.handsRadius;
        hands.offset = objectDef.handsOffset;
        hands.rotationDeg = objectDef.handsRotationDeg;
        sceneDef = sceneFromObjectDef(
            modelPath,
            objectDef.base, objectDef.normal,
            objectDef.roughness, objectDef.metallic,
            hands);
    }

    if (sceneDef.objects.empty())
    {
        fprintf(stderr, "Scene has no objects.\n");
        return 1;
    }

    // --- Load models ----------------------------------------------------------
    // Primary model (first object in scene) is loaded first.
    std::vector<Model> models(sceneDef.objects.size());
    for (size_t i = 0; i < sceneDef.objects.size(); ++i)
    {
        if (!loadModel(sceneDef.objects[i].modelPath.c_str(), models[i]))
        {
            fprintf(stderr, "Failed to load model: %s\n",
                    sceneDef.objects[i].modelPath.c_str());
            return 1;
        }
    }

    // Optional hand mesh.
    std::optional<Model> handModel;
    if (!sceneDef.hands.modelPath.empty())
    {
        Model hm;
        if (loadModel(sceneDef.hands.modelPath.c_str(), hm))
            handModel = std::move(hm);
        else
            SDL_Log("Hand model '%s' failed to load; using proxy fallback.",
                    sceneDef.hands.modelPath.c_str());
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

    XrSystem xr;
    const bool vrAvailable = xr.create();
    PlayerRig player;
    const glm::vec3 lightDir = glm::normalize(glm::vec3(-0.3f, -1.0f, -0.3f));

    try
    {
        Renderer renderer(window, vrAvailable ? &xr : nullptr);

        // Upload primary object (slot 0).
        const EntityDef &mainDef = sceneDef.objects[0];
        TextureSet mainTex{mainDef.texBase, mainDef.texNormal,
                           mainDef.texRoughness, mainDef.texMetallic};
        renderer.uploadObject(models[0], mainTex);

        // Upload additional meshes (slots 1+).
        std::vector<uint32_t> meshIds(sceneDef.objects.size(), 0u);
        for (size_t i = 1; i < sceneDef.objects.size(); ++i)
        {
            const EntityDef &def = sceneDef.objects[i];
            TextureSet tex{def.texBase, def.texNormal,
                           def.texRoughness, def.texMetallic};
            meshIds[i] = renderer.uploadMesh(models[i], tex);
        }

        // Hand mesh slot (if loaded).
        uint32_t handMeshId = 0; // default: reuse slot 0 (proxy fallback)
        if (handModel)
        {
            TextureSet emptyTex;
            handMeshId = renderer.uploadMesh(*handModel, emptyTex);
        }

        // --- Physics ----------------------------------------------------------
        PhysicsWorld physicsWorld(sceneDef.gravityY);
        float physicsAccumulator = 0.0f;

        // --- Build runtime entities -------------------------------------------
        static constexpr float kObjectDesiredRadius = 0.5f;
        static constexpr float kGrabRange = 0.35f;
        static constexpr float kObjectSpawnDistance = 1.5f;
        static constexpr float kObjectSpawnVerticalOffset = -0.25f;
        static constexpr int kXrStartupPlacementFrames = 45;

        std::vector<Entity> entities;
        entities.reserve(sceneDef.objects.size());
        for (size_t i = 0; i < sceneDef.objects.size(); ++i)
        {
            const EntityDef &def = sceneDef.objects[i];
            Entity e;
            e.id = def.id;
            e.meshId = meshIds[i];
            e.center = models[i].center;
            e.grabbable = def.grabbable;
            e.visible = true;
            e.rigidbody = def.rigidbody;

            // Compute spawn scale: normalise bounding sphere to kObjectDesiredRadius.
            const float s = models[i].radius > 0.0001f
                                ? kObjectDesiredRadius / models[i].radius
                                : 1.0f;
            e.transform.position = def.position;
            e.transform.rotation = glm::quat(glm::radians(def.rotationDeg));
            e.transform.scale = glm::vec3(def.scale * s);

            // Register physics body if component present.
            if (e.rigidbody && e.rigidbody->enabled)
                physicsWorld.addBody(e, *e.rigidbody,
                                     e.collider ? &*e.collider : nullptr,
                                     &models[i]);

            entities.push_back(std::move(e));
        }

        // Hand indicator entities (indices entities.size(), +1).
        const size_t kLeftHandIdx = entities.size();
        const size_t kRightHandIdx = entities.size() + 1;
        const HandsDef &hands = sceneDef.hands;
        const float handScale = (models[0].radius > 0.0001f && hands.enabled)
                                    ? hands.radius / models[0].radius
                                    : 0.0f;
        for (int hi = 0; hi < 2; ++hi)
        {
            Entity hEnt;
            hEnt.id = (hi == 0) ? "hand_left" : "hand_right";
            hEnt.meshId = handMeshId;
            hEnt.center = handModel ? handModel->center : models[0].center;
            hEnt.visible = false;
            hEnt.grabbable = false;
            hEnt.transform.position = {0.0f, -1000.0f, 0.0f};
            hEnt.transform.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            hEnt.transform.scale = glm::vec3(handScale);
            entities.push_back(std::move(hEnt));
        }

        // Kinematic hand physics colliders (sphere).
        void *handPhysicsBody[2] = {nullptr, nullptr};
        if (hands.enabled)
        {
            handPhysicsBody[0] = physicsWorld.addKinematicSphere(hands.radius);
            handPhysicsBody[1] = physicsWorld.addKinematicSphere(hands.radius);
        }

        int xrPlacementFramesRemaining =
            vrAvailable ? kXrStartupPlacementFrames : 0;

        // Grab state (tracks one grabbed object at a time).
        int grabbedEntityIdx = -1;
        int grabbingHandIdx = -1;
        glm::vec3 grabHandOffset{0.0f};

        bool showHandIndicators = true;
        std::array<glm::vec3, 2> handIndicatorPos{
            glm::vec3(0.0f), glm::vec3(0.0f)};
        std::array<glm::quat, 2> handIndicatorRot{
            glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
            glm::quat(1.0f, 0.0f, 0.0f, 0.0f)};
        std::array<bool, 2> handIndicatorValid{false, false};

        const glm::quat handRotOffset =
            glm::quat(glm::radians(hands.rotationDeg));

        auto buildDrawCalls = [&](bool xrRunning) -> std::vector<DrawCall>
        {
            std::vector<DrawCall> calls;
            calls.reserve(entities.size());
            for (size_t i = 0; i < entities.size(); ++i)
            {
                const auto &e = entities[i];
                // Hand indicator entities are handled below.
                if (i == kLeftHandIdx || i == kRightHandIdx)
                    continue;
                if (e.visible)
                    calls.push_back({e.meshId, entityMatrix(e)});
            }
            if (xrRunning && showHandIndicators && hands.enabled)
            {
                for (int hi = 0; hi < 2; ++hi)
                {
                    size_t idx = (hi == 0) ? kLeftHandIdx : kRightHandIdx;
                    if (handIndicatorValid[hi])
                    {
                        entities[idx].visible = true;
                        entities[idx].transform.position =
                            handIndicatorPos[hi] +
                            handIndicatorRot[hi] * hands.offset;
                        entities[idx].transform.rotation =
                            handIndicatorRot[hi] * handRotOffset;
                        calls.push_back({entities[idx].meshId,
                                         entityMatrix(entities[idx])});
                    }
                    else
                    {
                        entities[idx].visible = false;
                    }
                }
            }
            return calls;
        };

        renderer.setDrawCalls(buildDrawCalls(false));

        bool quit = false, dragging = false;
        SDL_Event e;

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

        float lodRatio = 1.0f, lodError = 0.02f;
        bool pendingLod = false;
        Model lodModel;
        bool leftGrabGesture = false, rightGrabGesture = false;

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
                    renderer.updateMesh(models[0]);
                else
                {
                    simplifyModel(models[0], lodRatio, lodError, lodModel);
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
                        player.snapTurn(-1.0f);
                    else if (e.key.key == SDLK_E)
                        player.snapTurn(1.0f);
                    break;
                case SDL_EVENT_WINDOW_RESIZED:
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                    renderer.onResize();
                    break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                    if (e.button.button == SDL_BUTTON_LEFT && !ImGui::GetIO().WantCaptureMouse)
                        dragging = true;
                    break;
                case SDL_EVENT_MOUSE_BUTTON_UP:
                    if (e.button.button == SDL_BUTTON_LEFT)
                        dragging = false;
                    break;
                case SDL_EVENT_MOUSE_MOTION:
                    if (dragging)
                    {
                        player.addYaw(e.motion.xrel * 0.0035f);
                        player.addPitch(-e.motion.yrel * 0.0035f);
                    }
                    break;
                default:
                    break;
                }
            }

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

            // --- Physics step -------------------------------------------------
            physicsAccumulator += dt;
            while (physicsAccumulator >= PhysicsWorld::kFixedStep)
            {
                physicsWorld.step(PhysicsWorld::kFixedStep);
                physicsAccumulator -= PhysicsWorld::kFixedStep;
            }
            // Write simulated positions back to entities (skips grabbed/static).
            physicsWorld.syncTransforms(entities);

            const bool xrRunning = vrAvailable && xr.sessionRunning();
            if (xrRunning)
            {
                xr.syncActions();
                leftGrabGesture = xr.isGrabGesture(XrSystem::Hand::Left);
                rightGrabGesture = xr.isGrabGesture(XrSystem::Hand::Right);
                player.moveLocal(xr.moveInput(), dt);
                player.snapTurn(xr.turnInput());

                renderer.setDrawCalls(buildDrawCalls(true));
                renderer.renderXrFrame(xr, player, lightDir);

                // Startup placement for first entity.
                if (grabbedEntityIdx < 0 && xrPlacementFramesRemaining > 0)
                {
                    entities[0].transform.position = placeInFrontOfEye(
                        renderer.lastEyePos, renderer.lastEyeView,
                        kObjectSpawnDistance, kObjectSpawnVerticalOffset);
                    --xrPlacementFramesRemaining;
                }

                // Hand pose extraction.
                auto getHandPose = [&](XrSystem::Hand h,
                                       glm::vec3 &pos,
                                       glm::quat &rot) -> bool
                {
                    const auto &jd = xr.getHandJoints(h);
                    if (jd.isTracked)
                    {
                        const auto &palm = jd.joints[XR_HAND_JOINT_PALM_EXT];
                        const auto flags = palm.locationFlags;
                        if (flags & XR_SPACE_LOCATION_POSITION_VALID_BIT)
                        {
                            pos = {palm.pose.position.x, palm.pose.position.y,
                                   palm.pose.position.z};
                            if (flags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)
                                rot = glm::quat(palm.pose.orientation.w,
                                                palm.pose.orientation.x,
                                                palm.pose.orientation.y,
                                                palm.pose.orientation.z);
                            return true;
                        }
                    }
                    bool valid = false;
                    const XrPosef grip = xr.handPose(h, valid);
                    if (valid)
                    {
                        pos = {grip.position.x, grip.position.y, grip.position.z};
                        rot = glm::quat(grip.orientation.w, grip.orientation.x,
                                        grip.orientation.y, grip.orientation.z);
                        return true;
                    }
                    return false;
                };

                // --- Grab logic -----------------------------------------------
                if (grabbedEntityIdx < 0)
                {
                    for (int hi = 0; hi < 2; ++hi)
                    {
                        glm::vec3 hpos{0.0f};
                        glm::quat hrot{1.0f, 0.0f, 0.0f, 0.0f};
                        if (!getHandPose(static_cast<XrSystem::Hand>(hi), hpos, hrot))
                            continue;
                        handIndicatorPos[hi] = hpos;
                        handIndicatorRot[hi] = hrot;
                        handIndicatorValid[hi] = true;
                        if (hands.enabled)
                            physicsWorld.moveKinematic(handPhysicsBody[hi], hpos, hrot);

                        if (!xr.isGrabGesture(static_cast<XrSystem::Hand>(hi)))
                            continue;

                        // Find nearest grabbable entity.
                        for (int ei = 0; ei < static_cast<int>(kLeftHandIdx); ++ei)
                        {
                            if (!entities[ei].grabbable)
                                continue;
                            if (glm::distance(hpos, entities[ei].transform.position) < kGrabRange)
                            {
                                grabbedEntityIdx = ei;
                                grabbingHandIdx = hi;
                                grabHandOffset = entities[ei].transform.position - hpos;
                                // Switch to kinematic if physics-driven.
                                if (entities[ei].physicsBody)
                                {
                                    auto *body = static_cast<btRigidBody *>(entities[ei].physicsBody);
                                    body->setCollisionFlags(
                                        body->getCollisionFlags() |
                                        btCollisionObject::CF_KINEMATIC_OBJECT);
                                    body->clearForces();
                                    btVector3 zero(0, 0, 0);
                                    body->setLinearVelocity(zero);
                                    body->setAngularVelocity(zero);
                                }
                                break;
                            }
                        }
                        if (grabbedEntityIdx >= 0)
                            break;
                    }
                }
                else
                {
                    glm::vec3 hpos{0.0f};
                    glm::quat hrot{1.0f, 0.0f, 0.0f, 0.0f};
                    const bool stillGrabbing =
                        getHandPose(static_cast<XrSystem::Hand>(grabbingHandIdx), hpos, hrot) &&
                        xr.isGrabGesture(static_cast<XrSystem::Hand>(grabbingHandIdx));
                    if (!stillGrabbing)
                    {
                        // Release: restore dynamic if applicable.
                        auto &ent = entities[grabbedEntityIdx];
                        if (ent.physicsBody)
                        {
                            auto *body = static_cast<btRigidBody *>(ent.physicsBody);
                            body->setCollisionFlags(
                                body->getCollisionFlags() &
                                ~btCollisionObject::CF_KINEMATIC_OBJECT);
                            body->activate(true);
                        }
                        grabbedEntityIdx = -1;
                        grabbingHandIdx = -1;
                    }
                    else
                    {
                        entities[grabbedEntityIdx].transform.position =
                            hpos + grabHandOffset;
                        if (entities[grabbedEntityIdx].physicsBody)
                            physicsWorld.moveKinematic(
                                entities[grabbedEntityIdx].physicsBody, hpos + grabHandOffset,
                                entities[grabbedEntityIdx].transform.rotation);
                        handIndicatorPos[grabbingHandIdx] = hpos;
                        handIndicatorRot[grabbingHandIdx] = hrot;
                    }
                }

                // Refresh hand indicators.
                for (int hi = 0; hi < 2; ++hi)
                {
                    if (hi == grabbingHandIdx)
                        continue;
                    glm::vec3 hpos{0.0f};
                    glm::quat hrot{1.0f, 0.0f, 0.0f, 0.0f};
                    handIndicatorValid[hi] =
                        getHandPose(static_cast<XrSystem::Hand>(hi), hpos, hrot);
                    if (handIndicatorValid[hi])
                    {
                        handIndicatorPos[hi] = hpos;
                        handIndicatorRot[hi] = hrot;
                        if (hands.enabled)
                            physicsWorld.moveKinematic(handPhysicsBody[hi], hpos, hrot);
                    }
                }
            }
            else
            {
                leftGrabGesture = rightGrabGesture = false;
                handIndicatorValid = {false, false};
            }

            renderer.setDrawCalls(buildDrawCalls(xrRunning));

            // --- ImGui --------------------------------------------------------
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
            if (ImGui::Combo("Anti-Aliasing", &msaaSelected, msaaLabels, msaaOptionCount))
                pendingMsaa = msaaOptions[msaaSelected].count;

            ImGui::SeparatorText("Mesh / LOD");
            ImGui::SliderFloat("Simplify", &lodRatio, 0.05f, 1.0f, "%.2f");
            ImGui::SliderFloat("Error", &lodError, 0.0f, 0.20f, "%.2f");
            if (ImGui::Button("Apply LOD"))
                pendingLod = true;
            ImGui::Text("Tris: %zu -> %u", models[0].indices.size() / 3,
                        renderer.getIndexCount() / 3);

            ImGui::SeparatorText("Player");
            ImGui::Text("Pos: %.2f %.2f %.2f", player.position.x,
                        player.position.y, player.position.z);

            // Per-entity transform UI.
            for (size_t i = 0; i < kLeftHandIdx; ++i)
            {
                auto &ent = entities[i];
                ImGui::PushID(static_cast<int>(i));
                const std::string label = "Object: " + ent.id;
                ImGui::SeparatorText(label.c_str());
                glm::vec3 p = ent.transform.position;
                if (ImGui::DragFloat3("Position", &p.x, 0.01f))
                    ent.transform.position = p;
                glm::vec3 euler = glm::degrees(glm::eulerAngles(ent.transform.rotation));
                if (ImGui::DragFloat3("Rotation (deg)", &euler.x, 0.5f))
                    ent.transform.rotation = glm::quat(glm::radians(euler));
                glm::vec3 sc = ent.transform.scale;
                if (ImGui::DragFloat3("Scale", &sc.x, 0.01f, 0.01f, 10.0f))
                    ent.transform.scale = sc;
                if (ent.rigidbody)
                {
                    ImGui::Text("Physics: %s",
                                ent.rigidbody->bodyType == BodyType::Dynamic ? "dynamic" : ent.rigidbody->bodyType == BodyType::Kinematic ? "kinematic"
                                                                                                                                          : "static");
                }
                ImGui::PopID();
            }

            ImGui::Checkbox("Show hand indicators", &showHandIndicators);

            if (xrRunning)
            {
                ImGui::SeparatorText("Grab");
                ImGui::Text("Hand tracking: %s",
                            xr.handTrackingAvailable() ? "yes" : "no");
                ImGui::Text("Left gesture:  %s", leftGrabGesture ? "yes" : "no");
                ImGui::Text("Right gesture: %s", rightGrabGesture ? "yes" : "no");
                ImGui::Text("Grabbed: %s",
                            grabbedEntityIdx >= 0
                                ? entities[grabbedEntityIdx].id.c_str()
                                : "none");
            }

            ImGui::End();
            ImGui::Render();

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
                float aspect = h > 0 ? static_cast<float>(w) / h : 1.0f;
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
