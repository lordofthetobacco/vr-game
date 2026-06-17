#include "include/model.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <meshoptimizer.h>

bool loadModel(const char *path, Model &out)
{
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(
        path,
        aiProcess_Triangulate |
            aiProcess_GenSmoothNormals |
            aiProcess_CalcTangentSpace |
            aiProcess_JoinIdenticalVertices |
            aiProcess_PreTransformVertices |
            aiProcess_ImproveCacheLocality |
            aiProcess_OptimizeMeshes |
            aiProcess_OptimizeMeshes |
            aiProcess_RemoveRedundantMaterials);

    if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) ||
        !scene->mRootNode)
    {
        fprintf(stderr, "Failed to load model '%s': %s\n", path,
                importer.GetErrorString());
        return false;
    }

    out.vertices.clear();
    out.indices.clear();

    glm::vec3 minB(std::numeric_limits<float>::max());
    glm::vec3 maxB(std::numeric_limits<float>::lowest());

    // aiProcess_PreTransformVertices bakes node transforms into the meshes, so
    // we can simply concatenate every mesh into one vertex/index buffer.
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m)
    {
        const aiMesh *mesh = scene->mMeshes[m];
        const uint32_t baseVertex = static_cast<uint32_t>(out.vertices.size());

        for (unsigned int v = 0; v < mesh->mNumVertices; ++v)
        {
            Vertex vert{};
            vert.pos = {mesh->mVertices[v].x, mesh->mVertices[v].y,
                        mesh->mVertices[v].z};

            if (mesh->HasNormals())
            {
                vert.normal = {mesh->mNormals[v].x, mesh->mNormals[v].y,
                               mesh->mNormals[v].z};
            }
            else
            {
                vert.normal = {0.0f, 1.0f, 0.0f};
            }

            if (mesh->HasTextureCoords(0))
            {
                vert.uv = {mesh->mTextureCoords[0][v].x,
                           1.0f - mesh->mTextureCoords[0][v].y};
            }
            else
            {
                vert.uv = {0.0f, 0.0f};
            }

            if (mesh->HasTangentsAndBitangents())
            {
                const glm::vec3 t{mesh->mTangents[v].x, mesh->mTangents[v].y,
                                  mesh->mTangents[v].z};
                const glm::vec3 b{mesh->mBitangents[v].x, mesh->mBitangents[v].y,
                                  mesh->mBitangents[v].z};
                // Handedness: sign that reconstructs the bitangent as
                // w * cross(N, T) in the shader.
                const float w =
                    (glm::dot(glm::cross(vert.normal, t), b) < 0.0f) ? -1.0f
                                                                     : 1.0f;
                vert.tangent = glm::vec4(t, w);
            }
            else
            {
                vert.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            }

            out.vertices.push_back(vert);

            minB = glm::min(minB, vert.pos);
            maxB = glm::max(maxB, vert.pos);
        }

        for (unsigned int f = 0; f < mesh->mNumFaces; ++f)
        {
            const aiFace &face = mesh->mFaces[f];
            // Triangulated, so every face has 3 indices.
            for (unsigned int i = 0; i < face.mNumIndices; ++i)
            {
                out.indices.push_back(baseVertex + face.mIndices[i]);
            }
        }
    }

    if (out.vertices.empty())
    {
        fprintf(stderr, "Model '%s' contains no vertices\n", path);
        return false;
    }

    out.center = 0.5f * (minB + maxB);
    out.radius = 0.5f * glm::length(maxB - minB);
    if (out.radius <= 0.0f)
    {
        out.radius = 1.0f;
    }

    printf("Loaded '%s': %zu vertices, %zu indices, radius %.3f\n", path,
           out.vertices.size(), out.indices.size(), out.radius);

    return true;
}

void simplifyModel(const Model &src, float ratio, float targetError, Model &out)
{
    out.center = src.center;
    out.radius = src.radius;

    const size_t srcIndexCount = src.indices.size();
    if (srcIndexCount == 0 || src.vertices.empty())
    {
        out.vertices = src.vertices;
        out.indices = src.indices;
        return;
    }

    // Target index count, clamped to a valid triangle multiple.
    ratio = std::clamp(ratio, 0.0f, 1.0f);
    size_t target =
        static_cast<size_t>(static_cast<double>(srcIndexCount) * ratio);
    target -= target % 3;
    if (target < 3) target = 3;

    // 1. Collapse the index buffer. meshopt only rewrites indices; the vertex
    // array stays the same (unused vertices are compacted away in step 3).
    std::vector<unsigned int> lod(srcIndexCount);
    float resultError = 0.0f;
    size_t lodCount = meshopt_simplify(
        lod.data(), src.indices.data(), srcIndexCount, &src.vertices[0].pos.x,
        src.vertices.size(), sizeof(Vertex), target, targetError,
        /*options*/ 0, &resultError);
    lod.resize(lodCount);

    // 2. Reorder for the post-transform vertex cache.
    meshopt_optimizeVertexCache(lod.data(), lod.data(), lodCount,
                                src.vertices.size());

    // 3. Compact to only the vertices the simplified mesh still references and
    // rewrite the indices to match (improves vertex-fetch locality + shrinks
    // the vertex buffer).
    out.indices = lod;
    out.vertices.resize(src.vertices.size());
    size_t vertexCount = meshopt_optimizeVertexFetch(
        out.vertices.data(), out.indices.data(), lodCount, src.vertices.data(),
        src.vertices.size(), sizeof(Vertex));
    out.vertices.resize(vertexCount);

    printf("Simplified: %zu -> %zu indices (%.0f%%), %zu vertices, error %.3f\n",
           srcIndexCount, lodCount,
           srcIndexCount ? 100.0 * static_cast<double>(lodCount) /
                               static_cast<double>(srcIndexCount)
                         : 0.0,
           vertexCount, resultError);
}
