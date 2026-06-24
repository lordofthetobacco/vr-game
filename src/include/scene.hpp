#ifndef SCENE_HPP
#define SCENE_HPP

#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "object.hpp" // Transform, transformToMatrix

// ---------------------------------------------------------------------------
// Collider component — collision shape, computed once before the game runs
// ---------------------------------------------------------------------------

enum class ColliderType : uint8_t
{
    Box,        // Explicit box using halfExtents
    MeshAABB,   // Box auto-fitted to the model's axis-aligned bounding box
    MeshConvex, // Convex hull built from all model vertex positions at startup
};

struct ColliderDef
{
    bool enabled = true;
    ColliderType type = ColliderType::MeshAABB;
    glm::vec3 halfExtents{0.1f, 0.1f, 0.1f}; // Box only; ignored otherwise
};

// ---------------------------------------------------------------------------
// Rigidbody component — simulation properties (shape comes from ColliderDef)
// ---------------------------------------------------------------------------

enum class BodyType : uint8_t
{
    Static,    // Never moves, infinite mass
    Dynamic,   // Fully simulated
    Kinematic, // Moved by code (grabbed / hand proxy)
};

struct RigidbodyDef
{
    bool enabled = true;
    BodyType bodyType = BodyType::Dynamic;
    float mass = 1.0f;
    float restitution = 0.3f;
    float friction = 0.5f;
    float linearDamping = 0.1f;
    float angularDamping = 0.1f;
    float gravityScale = 1.0f;
};

// ---------------------------------------------------------------------------
// Per-entity definition — loaded from [objects.name] dictionary-of-tables
// ---------------------------------------------------------------------------

struct EntityDef
{
    std::string id;
    std::string modelPath;

    std::string texBase;
    std::string texNormal;
    std::string texRoughness;
    std::string texMetallic;

    glm::vec3 position{0.0f, 1.2f, -1.5f};
    glm::vec3 rotationDeg{0.0f};
    float scale = 1.0f;
    bool grabbable = true;

    // Optional components
    std::optional<ColliderDef> collider;
    std::optional<RigidbodyDef> rigidbody;
};

// ---------------------------------------------------------------------------
// Hands configuration
// ---------------------------------------------------------------------------

struct HandsDef
{
    bool enabled = true;
    float radius = 0.06f;
    std::string modelPath; // empty = use proxy (same mesh scaled down)
    glm::vec3 offset{0.0f};
    glm::vec3 rotationDeg{0.0f};
};

// ---------------------------------------------------------------------------
// Full scene definition loaded from a scene .toml file
// ---------------------------------------------------------------------------

struct SceneDef
{
    float gravityY = -9.81f;
    HandsDef hands;
    std::vector<EntityDef> objects;
};

// ---------------------------------------------------------------------------
// Runtime entity (lives in the scene vector)
// ---------------------------------------------------------------------------

struct Entity
{
    std::string id;
    Transform transform;
    glm::vec3 center{0.0f}; // model-space pivot offset
    uint32_t meshId = 0;    // index into Renderer mesh slots
    bool visible = true;
    bool grabbable = true;

    std::optional<ColliderDef> collider;
    std::optional<RigidbodyDef> rigidbody;

    void *physicsBody = nullptr; // btRigidBody* as void*
};

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

// Parses a scene TOML using the [objects.name] dictionary-of-tables format.
// Returns false on failure, printing errors to stderr.
bool loadSceneToml(const char *path, SceneDef &out);

// Synthesise a SceneDef from the legacy single-object command-line path.
SceneDef sceneFromObjectDef(const std::string &modelPath,
                            const std::string &base,
                            const std::string &normal,
                            const std::string &roughness,
                            const std::string &metallic,
                            const HandsDef &hands);

#endif // SCENE_HPP
