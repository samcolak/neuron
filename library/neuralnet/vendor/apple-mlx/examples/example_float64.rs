#![allow(unsafe_op_in_unsafe_fn)]

#[path = "support/mod.rs"]
mod support;

use apple_mlx::raw;
use support::{Result, check, print_array};

fn main() -> Result<()> {
    unsafe {
        let stream = raw::mlx_default_cpu_stream_new();
        let data = [1f64, 2., 3., 4., 5., 6.];
        let shape = [2, 3];
        let mut arr = raw::mlx_array_new_data(
            data.as_ptr().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            raw::mlx_dtype__MLX_FLOAT64,
        );
        print_array("hello world in float64!", arr)?;

        let three = raw::mlx_array_new_float64(3.0);
        print_array("a float64 scalar array", three)?;
        check(
            raw::mlx_multiply(&mut arr, arr, three, stream),
            "mlx_multiply",
        )?;
        print_array("multiply previous array by 3!", arr)?;

        let two = raw::mlx_array_new_int(2);
        check(raw::mlx_divide(&mut arr, arr, two, stream), "mlx_divide")?;
        print_array("divive by 2 (integer)", arr)?;

        check(
            raw::mlx_arange(&mut arr, 0.0, 3.0, 0.5, raw::mlx_dtype__MLX_FLOAT64, stream),
            "mlx_arange",
        )?;
        print_array("arange", arr)?;

        let _ = raw::mlx_array_free(arr);
        let _ = raw::mlx_array_free(two);
        let _ = raw::mlx_array_free(three);
        let _ = raw::mlx_stream_free(stream);
    }
    Ok(())
}
