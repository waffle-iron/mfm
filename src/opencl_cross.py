from __future__ import absolute_import, print_function
import pyopencl


ctx = pyopencl.create_some_context()
queue = pyopencl.CommandQueue(ctx)

mf = pyopencl.mem_flags
prg = pyopencl.Program(ctx, """
__kernel void crs(__global const float4 *a_g, __global const float4 *b_g,
                  __global float4 *res_g) {
    int gid = get_global_id(0);
    res_g[gid] = cross(a_g[gid], b_g[gid]);
}
__kernel void nrm(__global float4 *res_g) {
    int gid = get_global_id(0);
    res_g[gid] = normalize(res_g[gid]);
}
""").build()
