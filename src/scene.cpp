#include "include/scene.hpp"

#define TOML_HEADER_ONLY 1
#include "third_party/toml.hpp"

#include <cstdio>
#include <glm/gtc/quaternion.hpp>
#include <string>

namespace
{
    std::string resolvePath(const std::string &baseDir, const std::string &p)
    {
        if (p.empty() || p.front() == '/')
            return p;
        return baseDir + p;
    }

    std::string dirOf(const std::string &path)
    {
        size_t slash = path.find_last_of("/\\");
        return slash == std::string::npos ? "" : path.substr(0, slash + 1);
    }

    ColliderDef parseCollider(const toml::table &c)
    {
        ColliderDef def;
        def.enabled = c["enabled"].value_or(true);

        std::string t = c["type"].value_or(std::string("mesh_aabb"));
        if (t == "box")
            def.type = ColliderType::Box;
        else if (t == "mesh_convex")
            def.type = ColliderType::MeshConvex;
        else
            def.type = ColliderType::MeshAABB;

        if (auto *he = c["half_extents"].as_array(); he && he->size() == 3)
        {
            def.halfExtents = {
                static_cast<float>(he->get(0)->value_or(0.1)),
                static_cast<float>(he->get(1)->value_or(0.1)),
                static_cast<float>(he->get(2)->value_or(0.1))};
        }
        return def;
    }

    RigidbodyDef parseRigidbody(const toml::table &rb)
    {
        RigidbodyDef def;
        def.enabled = rb["enabled"].value_or(true);

        std::string bt = rb["body_type"].value_or(std::string("dynamic"));
        if (bt == "static")
            def.bodyType = BodyType::Static;
        else if (bt == "kinematic")
            def.bodyType = BodyType::Kinematic;
        else
            def.bodyType = BodyType::Dynamic;

        def.mass = static_cast<float>(rb["mass"].value_or(1.0));
        def.restitution = static_cast<float>(rb["restitution"].value_or(0.3));
        def.friction = static_cast<float>(rb["friction"].value_or(0.5));
        def.linearDamping = static_cast<float>(rb["linear_damping"].value_or(0.1));
        def.angularDamping = static_cast<float>(rb["angular_damping"].value_or(0.1));
        def.gravityScale = static_cast<float>(rb["gravity_scale"].value_or(1.0));
        return def;
    }

    HandsDef parseHands(const toml::table &h, const std::string &baseDir)
    {
        HandsDef def;
        def.enabled = h["enabled"].value_or(true);
        def.radius = static_cast<float>(h["radius"].value_or(0.06));
        def.modelPath = resolvePath(baseDir, h["model"].value_or(std::string{}));
        def.offset.x = static_cast<float>(h["offset_x"].value_or(0.0));
        def.offset.y = static_cast<float>(h["offset_y"].value_or(0.0));
        def.offset.z = static_cast<float>(h["offset_z"].value_or(0.0));
        def.rotationDeg.x = static_cast<float>(h["rotation_x"].value_or(0.0));
        def.rotationDeg.y = static_cast<float>(h["rotation_y"].value_or(0.0));
        def.rotationDeg.z = static_cast<float>(h["rotation_z"].value_or(0.0));
        return def;
    }

    bool parseEntity(const std::string &name, const toml::table &obj,
                     const std::string &baseDir, EntityDef &out)
    {
        out.id = name;
        out.modelPath = resolvePath(baseDir, obj["model"].value_or(std::string{}));
        if (out.modelPath.empty())
        {
            fprintf(stderr, "Scene object '%s' is missing required 'model' key\n",
                    name.c_str());
            return false;
        }
        out.grabbable = obj["grabbable"].value_or(true);
        out.scale = static_cast<float>(obj["scale"].value_or(1.0));

        if (auto *pos = obj["position"].as_array(); pos && pos->size() == 3)
            out.position = {static_cast<float>(pos->get(0)->value_or(0.0)),
                            static_cast<float>(pos->get(1)->value_or(0.0)),
                            static_cast<float>(pos->get(2)->value_or(0.0))};

        if (auto *rot = obj["rotation_deg"].as_array(); rot && rot->size() == 3)
            out.rotationDeg = {static_cast<float>(rot->get(0)->value_or(0.0)),
                               static_cast<float>(rot->get(1)->value_or(0.0)),
                               static_cast<float>(rot->get(2)->value_or(0.0))};

        if (auto *tx = obj["textures"].as_table())
        {
            out.texBase = resolvePath(baseDir, (*tx)["base"].value_or(std::string{}));
            out.texNormal = resolvePath(baseDir, (*tx)["normal"].value_or(std::string{}));
            out.texRoughness = resolvePath(baseDir, (*tx)["roughness"].value_or(std::string{}));
            out.texMetallic = resolvePath(baseDir, (*tx)["metallic"].value_or(std::string{}));
        }

        if (auto *col = obj["collider"].as_table())
            out.collider = parseCollider(*col);

        if (auto *rb = obj["rigidbody"].as_table())
            out.rigidbody = parseRigidbody(*rb);

        return true;
    }
} // namespace

bool loadSceneToml(const char *path, SceneDef &out)
{
    toml::table tbl;
    try
    {
        tbl = toml::parse_file(path);
    }
    catch (const toml::parse_error &err)
    {
        fprintf(stderr, "Scene TOML parse error in '%s': %s\n",
                path, err.description().data());
        return false;
    }

    const std::string baseDir = dirOf(path);

    // [config]
    if (auto *cfg = tbl["config"].as_table())
        out.gravityY = static_cast<float>((*cfg)["gravity_y"].value_or(-9.81));

    // [hands]
    if (auto *h = tbl["hands"].as_table())
        out.hands = parseHands(*h, baseDir);

    // [objects] — dictionary-of-tables: each key is the object's name
    // Format:
    //   [objects.lightsaber]
    //   model = "..."
    //   [objects.lightsaber.collider]
    //   [objects.lightsaber.rigidbody]
    const auto *objectsNode = tbl["objects"].as_table();
    if (!objectsNode || objectsNode->empty())
    {
        fprintf(stderr, "Scene TOML '%s': [objects] table is missing or empty\n", path);
        return false;
    }

    for (auto &&[key, value] : *objectsNode)
    {
        const toml::table *objTbl = value.as_table();
        if (!objTbl)
            continue;
        EntityDef e;
        if (!parseEntity(std::string(key.str()), *objTbl, baseDir, e))
            return false;
        out.objects.push_back(std::move(e));
    }

    if (out.objects.empty())
    {
        fprintf(stderr, "Scene TOML '%s': no objects defined\n", path);
        return false;
    }

    return true;
}

SceneDef sceneFromObjectDef(const std::string &modelPath,
                            const std::string &base,
                            const std::string &normal,
                            const std::string &roughness,
                            const std::string &metallic,
                            const HandsDef &hands)
{
    SceneDef scene;
    scene.hands = hands;

    EntityDef e;
    e.id = "main_object";
    e.modelPath = modelPath;
    e.texBase = base;
    e.texNormal = normal;
    e.texRoughness = roughness;
    e.texMetallic = metallic;
    e.grabbable = true;
    scene.objects.push_back(std::move(e));
    return scene;
}
