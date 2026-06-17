#ifndef CAMERA_HPP
#define CAMERA_HPP

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Orbit camera: always looks at `target`. Drag rotates (yaw/pitch), scroll
// zooms (changes distance). Initialise from a model's center/radius so the
// model is framed on load.
class OrbitCamera {
public:
    void frame(const glm::vec3 &center, float radius) {
        target = center;
        // Place the camera far enough to see the whole bounding sphere.
        distance = radius * 3.0f;
        minDistance = radius * 0.1f;
        maxDistance = radius * 20.0f;
        yaw = glm::radians(45.0f);
        pitch = glm::radians(20.0f);
    }

    void rotate(float dx, float dy) {
        yaw += dx * rotateSpeed;
        pitch += dy * rotateSpeed;
        // Avoid gimbal flip at the poles.
        const float limit = glm::radians(89.0f);
        pitch = std::clamp(pitch, -limit, limit);
    }

    void zoom(float delta) {
        // Multiplicative so zoom feels consistent at any distance.
        distance *= std::pow(0.9f, delta);
        distance = std::clamp(distance, minDistance, maxDistance);
    }

    glm::vec3 position() const {
        const float cp = std::cos(pitch);
        return target + distance * glm::vec3(cp * std::cos(yaw), std::sin(pitch),
                                             cp * std::sin(yaw));
    }

    glm::mat4 viewMatrix() const {
        return glm::lookAt(position(), target, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::mat4 projMatrix(float aspect) const {
        glm::mat4 proj =
            glm::perspective(glm::radians(45.0f), aspect, 0.01f, 1000.0f);
        // GLM is built for OpenGL; flip Y for Vulkan's clip space.
        proj[1][1] *= -1.0f;
        return proj;
    }

private:
    glm::vec3 target{0.0f};
    float distance{3.0f};
    float minDistance{0.1f};
    float maxDistance{60.0f};
    float yaw{glm::radians(45.0f)};
    float pitch{glm::radians(20.0f)};

    static constexpr float rotateSpeed = 0.01f;
};

#endif
