#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 lightDir; // world-space direction the light travels (from above)
    vec4 camPos;   // world-space camera position
} ubo;

layout(binding = 1) uniform sampler2D baseTex;
layout(binding = 2) uniform sampler2D normalTex;
layout(binding = 3) uniform sampler2D roughnessTex;
layout(binding = 4) uniform sampler2D metallicTex;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec4 fragTangent;
layout(location = 3) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// GGX / Trowbridge-Reitz normal distribution.
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

// Smith geometry with Schlick-GGX, direct-lighting k.
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float ggxV = geometrySchlickGGX(max(dot(N, V), 0.0), roughness);
    float ggxL = geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
    return ggxV * ggxL;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    // --- material samples ---
    vec3 albedo = texture(baseTex, fragUV).rgb; // sRGB texture -> linear
    float roughness = clamp(texture(roughnessTex, fragUV).g, 0.04, 1.0);
    float metallic = texture(metallicTex, fragUV).b;

    // --- normal mapping (tangent space -> world) ---
    vec3 N = normalize(fragNormal);
    vec3 T = normalize(fragTangent.xyz - N * dot(N, fragTangent.xyz));
    vec3 B = cross(N, T) * fragTangent.w;
    vec3 nTex = texture(normalTex, fragUV).xyz * 2.0 - 1.0;
    mat3 TBN = mat3(T, B, N);
    N = normalize(TBN * nTex);

    // Two-sided: culling is disabled, so flip the normal on back faces.
    if (!gl_FrontFacing) {
        N = -N;
    }

    vec3 V = normalize(ubo.camPos.xyz - fragWorldPos);
    // lightDir travels downward (from above); L points toward the light.
    vec3 L = normalize(-ubo.lightDir.xyz);
    vec3 H = normalize(V + L);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Cook-Torrance specular.
    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kd * albedo / PI;

    vec3 lightColor = vec3(3.0); // simple white directional light
    vec3 Lo = (diffuse + specular) * lightColor * NdotL;

    // Constant ambient so unlit faces are not pure black.
    vec3 ambient = vec3(0.08) * albedo;
    vec3 color = ambient + Lo;

    // Reinhard tonemap; sRGB swapchain applies the gamma encode on write.
    color = color / (color + vec3(1.0));

    outColor = vec4(color, 1.0);
}
