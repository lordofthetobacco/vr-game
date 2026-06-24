#ifndef OBJECT_HPP
#define OBJECT_HPP

#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// An object described by a .toml file: a model plus the four metallic-roughness
// PBR maps. All paths are resolved relative to the .toml file's directory.
struct ObjectDef
{
    std::string modelPath;
    std::string base;
    std::string normal;
    std::string roughness;
    std::string metallic;
    bool handsEnabled = true;
    float handsRadius = 0.06f;
    glm::vec3 handsOffset{0.0f};
    glm::vec3 handsRotationDeg{0.0f};
};

// Reusable transform component for scene objects.
struct Transform
{
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f};
};

// Build a model matrix from position, rotation and scale.
glm::mat4 transformToMatrix(const Transform &transform);

// Parses a .toml object definition. Returns false (with an error printed to
// stderr) if the file cannot be read or any required key is missing.
bool loadObjectToml(const char *path, ObjectDef &out);

#endif
