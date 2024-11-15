import warp as wp


def main():
    @wp.kernel
    def test_float_precision(large_float: float):
        orig = large_float
        modified = large_float + 1e-7

        wp.printf("%f + %f = %f\n", orig, 1e-7, modified)
        
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
    