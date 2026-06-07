#![allow(unsafe_op_in_unsafe_fn)]

#[path = "support/mod.rs"]
mod support;

use apple_mlx::raw;
use support::{Result, check, print_array, vector_array_get};

fn main() -> Result<()> {
    unsafe {
        let stream = raw::mlx_default_gpu_stream_new();
        let mut input = raw::mlx_array_new();
        let empty_key = raw::mlx_array {
            ctx: std::ptr::null_mut(),
        };
        let dims = [4, 16];
        check(
            raw::mlx_random_normal(
                &mut input,
                dims.as_ptr(),
                dims.len(),
                raw::mlx_dtype__MLX_FLOAT32,
                0.0,
                1.0,
                empty_key,
                stream,
            ),
            "mlx_random_normal",
        )?;

        let source = support::cstring(
            "uint elem = thread_position_in_grid.x;\
             T tmp = inp[elem];\
             out[elem] = metal::exp(tmp);",
        )?;
        let input_names = raw::mlx_vector_string_new_value(c"inp".as_ptr());
        let output_names = raw::mlx_vector_string_new_value(c"out".as_ptr());
        let kernel = raw::mlx_fast_metal_kernel_new(
            c"myexp".as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        );
        let config = raw::mlx_fast_metal_kernel_config_new();
        let inputs = raw::mlx_vector_array_new_value(input);
        check(
            raw::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                config,
                c"T".as_ptr(),
                raw::mlx_dtype__MLX_FLOAT32,
            ),
            "mlx_fast_metal_kernel_config_add_template_arg_dtype",
        )?;
        check(
            raw::mlx_fast_metal_kernel_config_set_grid(
                config,
                raw::mlx_array_size(input) as i32,
                1,
                1,
            ),
            "mlx_fast_metal_kernel_config_set_grid",
        )?;
        check(
            raw::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1),
            "mlx_fast_metal_kernel_config_set_thread_group",
        )?;
        check(
            raw::mlx_fast_metal_kernel_config_add_output_arg(
                config,
                raw::mlx_array_shape(input),
                raw::mlx_array_ndim(input),
                raw::mlx_array_dtype(input),
            ),
            "mlx_fast_metal_kernel_config_add_output_arg",
        )?;

        let mut outputs = raw::mlx_vector_array_new();
        check(
            raw::mlx_fast_metal_kernel_apply(&mut outputs, kernel, inputs, config, stream),
            "mlx_fast_metal_kernel_apply",
        )?;
        let output = vector_array_get(outputs, 0)?;

        print_array("input", input)?;
        print_array("output", output)?;

        let _ = raw::mlx_array_free(input);
        let _ = raw::mlx_array_free(output);
        let _ = raw::mlx_stream_free(stream);
        let _ = raw::mlx_fast_metal_kernel_config_free(config);
        let _ = raw::mlx_fast_metal_kernel_free(kernel);
        let _ = raw::mlx_vector_array_free(inputs);
        let _ = raw::mlx_vector_array_free(outputs);
        let _ = raw::mlx_vector_string_free(input_names);
        let _ = raw::mlx_vector_string_free(output_names);
    }
    Ok(())
}
