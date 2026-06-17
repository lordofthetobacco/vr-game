#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 lightDir; // world-space direction the light travels (from above)
    vec4 camPos;   // world-space camera position
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent; // xyz = tangent, w = handedness
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec4 fragTangent;
layout(location = 3) out vec2 fragUV;

void main() {
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;

    // Uniform-scale assumption: mat3(model) is fine for normal/tangent.
    mat3 normalMat = mat3(ubo.model);
    fragWorldPos = worldPos.xyz;
    fragNormal = normalMat * inNormal;
    fragTangent = vec4(normalMat * inTangent.xyz, inTangent.w);
    fragUV = inUV;
}
