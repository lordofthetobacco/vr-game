#ifndef PHYSICS_WORLD_HPP
#define PHYSICS_WORLD_HPP

#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// Forward declarations — no Bullet or scene headers needed here.
struct Entity;
struct RigidbodyDef;
struct ColliderDef;
struct Model;
class btDefaultCollisionConfiguration;
class btCollisionDispatcher;
class btDbvtBroadphase;
class btSequentialImpulseConstraintSolver;
class btDiscreteDynamicsWorld;
class btRigidBody;
class btCollisionShape;
class btMotionState;

// Thin wrapper around a btDiscreteDynamicsWorld.
// All Bullet types are hidden behind the PImpl pointer so translation units
// that include this header never see Bullet headers.
class PhysicsWorld
{
public:
    static constexpr float kDefaultGravityY = -9.81f;
    static constexpr float kFixedStep = 1.0f / 90.0f; // 90 Hz
    static constexpr int kMaxSubSteps = 4;

    explicit PhysicsWorld(float gravityY = kDefaultGravityY);
    ~PhysicsWorld();

    PhysicsWorld(const PhysicsWorld &) = delete;
    PhysicsWorld &operator=(const PhysicsWorld &) = delete;

    // Register a rigidbody for an entity.
    // `collider`  determines the collision shape (AABB/convex hull computed from
    //             `model` vertices before the game runs; Box uses explicit extents).
    // `model`     is required when collider type is MeshAABB or MeshConvex.
    // Sets entity.physicsBody.
    void addBody(Entity &entity, const RigidbodyDef &def,
                 const ColliderDef *collider = nullptr,
                 const Model *model = nullptr);

    // Create a kinematic sphere for a hand proxy (no mass, driven by pose).
    // Returns an opaque handle (btRigidBody*) cast to void*.
    void *addKinematicSphere(float radius);

    // Drive a kinematic body's target transform (e.g. hand proxies).
    void moveKinematic(void *body,
                       const glm::vec3 &pos,
                       const glm::quat &rot);

    // Advance simulation by `deltaTime` seconds (fixed sub-steps internally).
    void step(float deltaTime);

    // Copy simulated positions/rotations back into each entity's Transform.
    // Entities without a physicsBody are skipped.
    void syncTransforms(std::vector<Entity> &entities);

    // Remove a body and free its resources.
    void removeBody(void *body);

private:
    btDefaultCollisionConfiguration *m_collisionConfig = nullptr;
    btCollisionDispatcher *m_dispatcher = nullptr;
    btDbvtBroadphase *m_broadphase = nullptr;
    btSequentialImpulseConstraintSolver *m_solver = nullptr;
    btDiscreteDynamicsWorld *m_world = nullptr;

    // All shapes/motion states allocated here are owned by PhysicsWorld.
    std::vector<btCollisionShape *> m_shapes;
    std::vector<btMotionState *> m_motionStates;
    std::vector<btRigidBody *> m_bodies;

    btRigidBody *createBody(btCollisionShape *shape,
                            float mass,
                            const glm::vec3 &pos,
                            const glm::quat &rot,
                            float restitution,
                            float friction,
                            float linearDamping,
                            float angularDamping,
                            bool kinematic);
};

#endif // PHYSICS_WORLD_HPP
