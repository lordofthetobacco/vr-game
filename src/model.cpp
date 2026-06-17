#include "include/model.hpp"

#include <cstdio>
#include <limits>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

bool loadModel(const char *path, Model &out) {
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(
        path,
        aiProcess_Triangulate | aiProcess_GenSmoothNormals |
            aiProcess_CalcTangentSpace | aiProcess_JoinIdenticalVertices |
            aiProcess_PreTransformVertices);

    if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) ||
        !scene->mRootNode) {
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
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh *mesh = scene->mMeshes[m];
        const uint32_t baseVertex = static_cast<uint32_t>(out.vertices.size());

        for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
            Vertex vert{};
            vert.pos = {mesh->mVertices[v].x, mesh->mVertices[v].y,
                        mesh->mVertices[v].z};

            if (mesh->HasNormals()) {
                vert.normal = {mesh->mNormals[v].x, mesh->mNormals[v].y,
                               mesh->mNormals[v].z};
            } else {
                vert.normal = {0.0f, 1.0f, 0.0f};
            }

            if (mesh->HasTextureCoords(0)) {
                vert.uv = {mesh->mTextureCoords[0][v].x,
                           mesh->mTextureCoords[0][v].y};
            } else {
                vert.uv = {0.0f, 0.0f};
            }

            if (mesh->HasTangentsAndBitangents()) {
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
            } else {
                vert.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            }

            out.vertices.push_back(vert);

            minB = glm::min(minB, vert.pos);
            maxB = glm::max(maxB, vert.pos);
        }

        for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace &face = mesh->mFaces[f];
            // Triangulated, so every face has 3 indices.
            for (unsigned int i = 0; i < face.mNumIndices; ++i) {
                out.indices.push_back(baseVertex + face.mIndices[i]);
            }
        }
    }

    if (out.vertices.empty()) {
        fprintf(stderr, "Model '%s' contains no vertices\n", path);
        return false;
    }

    out.center = 0.5f * (minB + maxB);
    out.radius = 0.5f * glm::length(maxB - minB);
    if (out.radius <= 0.0f) {
        out.radius = 1.0f;
    }

    printf("Loaded '%s': %zu vertices, %zu indices, radius %.3f\n", path,
           out.vertices.size(), out.indices.size(), out.radius);

    return true;
}
