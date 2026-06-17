#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 lightDir;     // world-space direction the light travels (from above)
    vec4 camPos;       // world-space camera position
    vec4 quantMin;     // xyz: bbox min for position dequantization
    vec4 quantExtent;  // xyz: bbox extent (max - min)
} ubo;

// Quantized vertex inputs (see PackedVertex in renderer.cpp):
//   pos     R16G16B16A16_UNORM  -> [0,1] bbox-normalized, dequantized below
//   normal  R8G8B8A8_SNORM      -> [-1,1]
//   tangent R8G8B8A8_SNORM      -> xyz tangent, w = handedness sign
//   uv      R16G16_SFLOAT       -> already a float UV
layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec4 fragTangent;
layout(location = 3) out vec2 fragUV;

void main() {
    vec3 pos = inPosition.xyz * ubo.quantExtent.xyz + ubo.quantMin.xyz;
    vec4 worldPos = ubo.model * vec4(pos, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;

    // Uniform-scale assumption: mat3(model) is fine for normal/tangent.
    mat3 normalMat = mat3(ubo.model);
    fragWorldPos = worldPos.xyz;
    fragNormal = normalMat * normalize(inNormal.xyz);
    fragTangent = vec4(normalMat * normalize(inTangent.xyz), inTangent.w);
    fragUV = inUV;
}
