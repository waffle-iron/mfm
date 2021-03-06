#version 420

in vec3 mean_position;

out vec4 vertex_position;
out float shadow;

uniform mat4 light_matrix;
uniform mat4 rotation_matrix;

uniform float coefficients[199];
uniform int indices[199];
layout(binding=0) uniform sampler2D principal_components;
layout(binding=1) uniform sampler2DShadow depth_map;

void main(void) {
    int c_pos, i, j;
    ivec2 texPos;
    vec4 acc = vec4(0.0);
    for (i = 0; i < 199 && indices[i] > -1; i++) {
        j = indices[i];
        c_pos = gl_VertexID + 53490 * j;
        texPos = ivec2(c_pos % 8192, c_pos / 8192);
        acc += texelFetch(principal_components, texPos, 0) * coefficients[j];
    }
    vec4 source_position = vec4(mean_position + vec3(acc), 246006.0);
    gl_Position = rotation_matrix * source_position;
    vertex_position = gl_Position;

    vec4 light_projection = light_matrix * source_position;
    light_projection.xyz += light_projection.w;
    light_projection.w *= 2.0;

    shadow = 0.0;
    shadow += textureProjOffset(depth_map, light_projection, ivec2(-1, -1));
    shadow += textureProjOffset(depth_map, light_projection, ivec2(-1, 1));
    shadow += textureProjOffset(depth_map, light_projection, ivec2(1, -1));
    shadow += textureProjOffset(depth_map, light_projection, ivec2(1, 1));
    shadow += textureProjOffset(depth_map, light_projection, ivec2(0, 0));
    shadow = shadow < 3.0? shadow / 5.0 : 1.0;
}
