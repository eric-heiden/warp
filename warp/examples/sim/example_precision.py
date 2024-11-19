from contextlib import contextmanager

import warp as wp

@contextmanager
def force_float64_precision():
    # store originals
    original_float32 = wp.float32
    original_vec3 = wp.vec3
    original_mat22 = wp.mat22 
    original_mat33 = wp.mat33
    original_spatial_vector = wp.spatial_vector
    original_spatial_matrix = wp.spatial_matrix
    original_transform = wp.transform
    original_quat = wp.quat
    original_vec2 = wp.vec2

    try:
        # switch to float64
        wp.types.float32 = wp.float64
        wp.codegen.float32 = wp.float64
        wp.float32 = wp.float64
        wp.vec3 = wp.vec3d
        wp.mat22 = wp.mat22d
        wp.mat33 = wp.mat33d
        wp.spatial_vector = wp.spatial_vectord
        wp.spatial_matrix = wp.spatial_matrixd
        wp.transform = wp.transformd
        wp.quat = wp.quatd
        wp.vec2 = wp.vec2d
        # device = wp.context.runtime.get_device()
        # for _, module in wp.context.user_modules.items():
        #     module.unload()
        #     module.mark_modified()
        #     module.load(device)
        # wp.synchronize_device()
        yield
    finally:
        # restore
        wp.types.float32 = original_float32
        wp.codegen.float32 = original_float32
        wp.float32 = original_float32
        wp.vec3 = original_vec3
        wp.mat22 = original_mat22
        wp.mat33 = original_mat33
        wp.spatial_vector = original_spatial_vector
        wp.spatial_matrix = original_spatial_matrix
        wp.transform = original_transform
        wp.quat = original_quat
        wp.vec2 = original_vec2

def main():
    @wp.kernel
    def test_float_precision(large_float: float):
        orig = large_float
        modified = large_float + 1e-7

        wp.printf("%.10f + %.10f = %.10f\n", orig, 1e-7, modified)
        
        if orig == modified:
            wp.print("You do NOT appear to be doing f64 math\n")
        else:
            wp.print("You appear to be doing f64 math\n")

    wp.launch(
        kernel=test_float_precision,
        dim=1,
        inputs=[1e7]
    )

if __name__ == "__main__":
    wp.init()
    
    main()

    with force_float64_precision():
        main()
    
    main()
    