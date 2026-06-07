#![allow(unsafe_op_in_unsafe_fn)]

#[path = "support/mod.rs"]
mod support;

use apple_mlx::raw;
use support::{Result, check};

unsafe extern "C" {
    #[cfg(target_os = "macos")]
    static mut __stdoutp: *mut raw::FILE;
    #[cfg(not(target_os = "macos"))]
    static mut stdout: *mut raw::FILE;
}

unsafe fn stdout_file() -> *mut raw::FILE {
    #[cfg(target_os = "macos")]
    {
        __stdoutp
    }
    #[cfg(not(target_os = "macos"))]
    {
        stdout
    }
}

fn main() -> Result<()> {
    unsafe {
        let stream = raw::mlx_default_cpu_stream_new();
        let data = [1f32, 2., 3., 4., 5., 6.];
        let shape = [2, 3];
        let mut res = raw::mlx_array_new();
        let val = raw::mlx_array_new_data(
            data.as_ptr().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            raw::mlx_dtype__MLX_FLOAT32,
        );
        let two = raw::mlx_array_new_int(2);
        check(raw::mlx_divide(&mut res, val, two, stream), "mlx_divide")?;
        check(raw::mlx_log(&mut res, res, stream), "mlx_log")?;

        let namer = raw::mlx_node_namer_new();
        check(
            raw::mlx_node_namer_set_name(namer, val, c"inputs".as_ptr()),
            "mlx_node_namer_set_name inputs",
        )?;
        check(
            raw::mlx_node_namer_set_name(namer, res, c"result".as_ptr()),
            "mlx_node_namer_set_name result",
        )?;
        let vec = raw::mlx_vector_array_new();
        let _ = raw::mlx_vector_array_append_value(vec, val);
        let _ = raw::mlx_vector_array_append_value(vec, two);
        let _ = raw::mlx_vector_array_append_value(vec, res);

        check(
            raw::mlx_export_to_dot(stdout_file(), namer, vec),
            "mlx_export_to_dot",
        )?;

        let _ = raw::mlx_array_free(val);
        let _ = raw::mlx_array_free(two);
        let _ = raw::mlx_array_free(res);
        let _ = raw::mlx_vector_array_free(vec);
        let _ = raw::mlx_node_namer_free(namer);
        let _ = raw::mlx_stream_free(stream);
    }
    Ok(())
}
