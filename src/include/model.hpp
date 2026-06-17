#ifndef MODEL_HPP
#define MODEL_HPP

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec4 tangent; // xyz = tangent, w = bitangent handedness sign
    glm::vec2 uv;
};

struct Model {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // Bounding info used to frame the camera on load.
    glm::vec3 center{0.0f};
    float radius{1.0f};
};

// Loads an OBJ (or any assimp-supported) model. Returns false on failure.
bool loadModel(const char *path, Model &out);

// Produces a simplified + GPU-optimized copy of src into out. `ratio` (0,1] is
// the fraction of indices to keep; `targetError` is the max relative geometric
// error meshopt may introduce. Runs meshopt simplify + vertex-cache/fetch
// optimization. out.center/out.radius are carried over from src.
void simplifyModel(const Model &src, float ratio, float targetError, Model &out);

#endif
