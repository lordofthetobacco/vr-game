// physics_world.cpp is compiled with -w (warnings suppressed) because Bullet3
// headers produce many warnings under strict -Wall -Wextra -Wpedantic.

#include "include/physics_world.hpp"
#include "include/physics_bridge.hpp"
#include "include/scene.hpp"
#include "include/model.hpp"

#include <algorithm>
#include <limits>
#include <btBulletDynamicsCommon.h>
#include <BulletCollision/CollisionShapes/btConvexHullShape.h>
#include <stdexcept>

using namespace PhysicsBridge;

PhysicsWorld::PhysicsWorld(float gravityY)
{
    m_collisionConfig = new btDefaultCollisionConfiguration();
    m_dispatcher = new btCollisionDispatcher(m_collisionConfig);
    m_broadphase = new btDbvtBroadphase();
    m_solver = new btSequentialImpulseConstraintSolver();
    m_world = new btDiscreteDynamicsWorld(
        m_dispatcher, m_broadphase, m_solver, m_collisionConfig);
    m_world->setGravity(btVector3(0.0f, gravityY, 0.0f));
}

PhysicsWorld::~PhysicsWorld()
{
    // Remove all bodies first so the world doesn't reference freed memory.
    for (auto *body : m_bodies)
    {
        m_world->removeRigidBody(body);
        delete body;
    }
    for (auto *ms : m_motionStates)
        delete ms;
    for (auto *sh : m_shapes)
        delete sh;

    delete m_world;
    delete m_solver;
    delete m_broadphase;
    delete m_dispatcher;
    delete m_collisionConfig;
}

btRigidBody *PhysicsWorld::createBody(btCollisionShape *shape,
                                      float mass,
                                      const glm::vec3 &pos,
                                      const glm::quat &rot,
                                      float restitution,
                                      float friction,
                                      float linearDamping,
                                      float angularDamping,
                                      bool kinematic)
{
    btTransform startTransform;
    startTransform.setIdentity();
    startTransform.setOrigin(glmToBt(pos));
    startTransform.setRotation(glmToBt(rot));

    btVector3 localInertia(0.0f, 0.0f, 0.0f);
    if (mass > 0.0f && !kinematic)
        shape->calculateLocalInertia(mass, localInertia);

    auto *ms = new btDefaultMotionState(startTransform);
    float useMass = kinematic ? 0.0f : mass;
    btRigidBody::btRigidBodyConstructionInfo ci(useMass, ms, shape, localInertia);
    ci.m_restitution = restitution;
    ci.m_friction = friction;
    ci.m_linearDamping = linearDamping;
    ci.m_angularDamping = angularDamping;

    auto *body = new btRigidBody(ci);

    if (kinematic)
    {
        body->setCollisionFlags(body->getCollisionFlags() |
                                btCollisionObject::CF_KINEMATIC_OBJECT);
        body->setActivationState(DISABLE_DEACTIVATION);
    }

    m_shapes.push_back(shape);
    m_motionStates.push_back(ms);
    m_bodies.push_back(body);
    m_world->addRigidBody(body);
    return body;
}

void PhysicsWorld::addBody(Entity &entity, const RigidbodyDef &def,
                           const ColliderDef *collider, const Model *model)
{
    if (!def.enabled)
        return;

    // -----------------------------------------------------------------------
    // Build collision shape from ColliderDef (or fall back to unit sphere)
    // -----------------------------------------------------------------------
    btCollisionShape *shape = nullptr;

    if (collider && collider->enabled)
    {
        switch (collider->type)
        {
        case ColliderType::Box:
            shape = new btBoxShape(glmToBt(collider->halfExtents));
            break;

        case ColliderType::MeshAABB:
        {
            // Compute tight AABB from all vertex positions.
            glm::vec3 minB(std::numeric_limits<float>::max());
            glm::vec3 maxB(std::numeric_limits<float>::lowest());
            if (model && !model->vertices.empty())
            {
                for (const auto &v : model->vertices)
                {
                    minB = glm::min(minB, v.pos);
                    maxB = glm::max(maxB, v.pos);
                }
            }
            else
            {
                minB = glm::vec3(-0.1f);
                maxB = glm::vec3(0.1f);
            }
            const glm::vec3 he = (maxB - minB) * 0.5f;
            shape = new btBoxShape(glmToBt(glm::max(he, glm::vec3(0.001f))));
            break;
        }

        case ColliderType::MeshConvex:
        {
            auto *hull = new btConvexHullShape();
            if (model && !model->vertices.empty())
            {
                for (const auto &v : model->vertices)
                    hull->addPoint(glmToBt(v.pos), /*recalcAABB=*/false);
                hull->recalcLocalAabb();
            }
            else
            {
                // Degenerate fallback: unit cube.
                for (int x : {-1, 1})
                    for (int y : {-1, 1})
                        for (int z : {-1, 1})
                            hull->addPoint(btVector3(x * 0.1f, y * 0.1f, z * 0.1f));
            }
            shape = hull;
            break;
        }
        }
    }

    if (!shape)
        shape = new btSphereShape(0.1f); // safe fallback when no collider given

    bool kinematic = (def.bodyType == BodyType::Kinematic);
    float mass = (def.bodyType == BodyType::Static) ? 0.0f : def.mass;

    btRigidBody *body = createBody(
        shape, mass,
        entity.transform.position,
        entity.transform.rotation,
        def.restitution,
        def.friction,
        def.linearDamping,
        def.angularDamping,
        kinematic);

    if (def.bodyType == BodyType::Dynamic)
        body->setGravity(body->getGravity() * def.gravityScale);

    entity.physicsBody = static_cast<void *>(body);
}

void *PhysicsWorld::addKinematicSphere(float radius)
{
    auto *shape = new btSphereShape(radius);
    btRigidBody *body = createBody(
        shape, 0.0f, glm::vec3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        0.0f, 0.0f, 0.0f, 0.0f, /*kinematic=*/true);
    return static_cast<void *>(body);
}

void PhysicsWorld::moveKinematic(void *handle,
                                 const glm::vec3 &pos,
                                 const glm::quat &rot)
{
    if (!handle)
        return;
    auto *body = static_cast<btRigidBody *>(handle);
    btTransform t;
    t.setOrigin(glmToBt(pos));
    t.setRotation(glmToBt(rot));
    body->getMotionState()->setWorldTransform(t);
    body->setWorldTransform(t);
}

void PhysicsWorld::step(float deltaTime)
{
    m_world->stepSimulation(deltaTime, kMaxSubSteps, kFixedStep);
}

void PhysicsWorld::syncTransforms(std::vector<Entity> &entities)
{
    for (auto &entity : entities)
    {
        if (!entity.physicsBody)
            continue;
        auto *body = static_cast<btRigidBody *>(entity.physicsBody);

        // Kinematic and static bodies are driven externally; don't overwrite.
        if (body->isStaticOrKinematicObject())
            continue;

        btTransform t;
        body->getMotionState()->getWorldTransform(t);
        entity.transform.position = btToGlm(t.getOrigin());
        entity.transform.rotation = btToGlm(t.getRotation());
    }
}

void PhysicsWorld::removeBody(void *handle)
{
    if (!handle)
        return;
    auto *body = static_cast<btRigidBody *>(handle);
    m_world->removeRigidBody(body);

    if (body->getMotionState())
    {
        btMotionState *ms = body->getMotionState();
        m_motionStates.erase(
            std::remove(m_motionStates.begin(), m_motionStates.end(), ms),
            m_motionStates.end());
        delete ms;
    }
    m_bodies.erase(
        std::remove(m_bodies.begin(), m_bodies.end(), body),
        m_bodies.end());
    delete body;
}
