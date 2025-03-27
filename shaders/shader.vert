#version 450

<<<<<<< Updated upstream
=======
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

>>>>>>> Stashed changes
layout(location = 0) out vec3 fragColor;

void main() {
<<<<<<< Updated upstream
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
=======
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = inNormal * 0.5 + 0.5;
>>>>>>> Stashed changes
}