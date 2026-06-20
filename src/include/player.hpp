#ifndef PLAYER_HPP
#define PLAYER_HPP

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// The player's locomotion rig: a world-space position on the floor plane plus a
// heading (yaw). It defines the transform from the OpenXR stage origin into the
// world; the per-eye HMD pose is composed on top of this each frame, so the
// headset supplies head position/orientation while WASD / thumbstick drive the
// rig over the ground.
//
// In the desktop-only fallback (no HMD) the same rig doubles as a free-look FPS
// camera: pitch is then meaningful and `eyeView()` builds the view directly.
class PlayerRig {
public:
    // worldFromStage: translate to the rig position, then rotate by yaw about Y.
    glm::mat4 worldFromStage() const {
        glm::mat4 m = glm::translate(glm::mat4(1.0f), position);
        m = glm::rotate(m, yaw, glm::vec3(0.0f, 1.0f, 0.0f));
        return m;
    }

    // Move on the horizontal plane, relative to the current heading.
    // input.x = strafe (+right), input.y = forward (+forward).
    void moveLocal(glm::vec2 input, float dt) {
        if (input.x == 0.0f && input.y == 0.0f) return;
        const float s = std::sin(yaw), c = std::cos(yaw);
        // Forward (yaw=0) points toward -Z; right points toward +X.
        glm::vec3 forward(-s, 0.0f, -c);
        glm::vec3 right(c, 0.0f, -s);
        glm::vec3 delta = (right * input.x + forward * input.y) * moveSpeed * dt;
        position += delta;
    }

    // Snap-turn by a fixed step (dir = -1 left, +1 right). Edge-triggered by the
    // caller; this just applies one increment.
    void snapTurn(float dir) {
        if (dir == 0.0f) return;
        yaw -= dir * snapStep; // turning right (+) decreases yaw about +Y
    }

    // Smooth yaw for keyboard/mouse look (radians applied directly).
    void addYaw(float radians) { yaw -= radians; }
    void addPitch(float radians) {
        pitch = std::clamp(pitch + radians, -glm::radians(89.0f),
                           glm::radians(89.0f));
    }

    // Desktop free-look view matrix (uses yaw + pitch, no HMD).
    glm::mat4 eyeView() const {
        glm::vec3 eye = position + glm::vec3(0.0f, eyeHeight, 0.0f);
        const float cp = std::cos(pitch);
        glm::vec3 dir(-std::sin(yaw) * cp, std::sin(pitch),
                      -std::cos(yaw) * cp);
        return glm::lookAt(eye, eye + dir, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::vec3 eyePosition() const {
        return position + glm::vec3(0.0f, eyeHeight, 0.0f);
    }

    glm::vec3 position{0.0f, 0.0f, 0.0f};
    float yaw{0.0f};
    float pitch{0.0f};

    float moveSpeed{2.5f};                  // m/s
    float snapStep{glm::radians(30.0f)};    // per snap-turn
    float eyeHeight{1.6f};                  // desktop fallback eye height
};

#endif
