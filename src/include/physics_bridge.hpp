#ifndef PHYSICS_BRIDGE_HPP
#define PHYSICS_BRIDGE_HPP

// Inline helpers converting between GLM and Bullet3 math types.
// Include this header only from translation units that also include Bullet.

#include <btBulletDynamicsCommon.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace PhysicsBridge
{

    inline btVector3 glmToBt(const glm::vec3 &v)
    {
        return btVector3(v.x, v.y, v.z);
    }

    inline glm::vec3 btToGlm(const btVector3 &v)
    {
        return glm::vec3(v.getX(), v.getY(), v.getZ());
    }

    inline btQuaternion glmToBt(const glm::quat &q)
    {
        // Bullet: (x, y, z, w)
        return btQuaternion(q.x, q.y, q.z, q.w);
    }

    inline glm::quat btToGlm(const btQuaternion &q)
    {
        // GLM quat ctor: (w, x, y, z)
        return glm::quat(q.getW(), q.getX(), q.getY(), q.getZ());
    }

} // namespace PhysicsBridge

#endif // PHYSICS_BRIDGE_HPP
