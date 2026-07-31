#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

void main() {
    // Simple directional light (from upper-right-front).
    vec3 lightDir = normalize(vec3(0.5, 0.8, 0.3));
    float diff = max(dot(normalize(fragNormal), lightDir), 0.0);

    // Ambient + diffuse
    vec3 ambient = fragColor * 0.25;
    vec3 diffuse = fragColor * diff * 0.75;

    outColor = vec4(ambient + diffuse, 1.0);
}
