#version 300 es

in vec3 mean_position;

uniform mat4 light_matrix;

uniform float coefficients[199];
uniform int indices[199];
uniform sampler2D principal_components;

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
    gl_Position = light_matrix * vec4(mean_position + vec3(acc), 246006.0);
}
