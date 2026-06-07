#![allow(unsafe_op_in_unsafe_fn)]

#[path = "support/mod.rs"]
mod support;

use apple_mlx::raw;
use std::ffi::CStr;
use std::ptr;
use support::{Result, check, cstring, print_array, vector_string_get};

fn gpu_info() -> Result<()> {
    unsafe {
        println!("==================================================");
        println!("GPU info (using mlx_device_info API):");

        let mut gpu_count = 0;
        check(
            raw::mlx_device_count(&mut gpu_count, raw::mlx_device_type__MLX_GPU),
            "mlx_device_count",
        )?;
        println!("GPU device count: {gpu_count}");

        let mut dev = raw::mlx_device_new();
        check(
            raw::mlx_get_default_device(&mut dev),
            "mlx_get_default_device",
        )?;

        let mut info = raw::mlx_device_info_new();
        if raw::mlx_device_info_get(&mut info, dev) == 0 {
            let mut keys = raw::mlx_vector_string_new();
            check(
                raw::mlx_device_info_get_keys(&mut keys, info),
                "mlx_device_info_get_keys",
            )?;
            let num_keys = raw::mlx_vector_string_size(keys);
            println!("Device info ({num_keys} keys):");
            for i in 0..num_keys {
                let key = vector_string_get(keys, i)?;
                let key_c = cstring(&key)?;
                let mut is_string = false;
                check(
                    raw::mlx_device_info_is_string(&mut is_string, info, key_c.as_ptr()),
                    "mlx_device_info_is_string",
                )?;
                if is_string {
                    let mut value = ptr::null();
                    check(
                        raw::mlx_device_info_get_string(&mut value, info, key_c.as_ptr()),
                        "mlx_device_info_get_string",
                    )?;
                    let rendered = if value.is_null() {
                        "(null)".to_string()
                    } else {
                        CStr::from_ptr(value).to_string_lossy().into_owned()
                    };
                    println!("  {key}: {rendered}");
                } else {
                    let mut value = 0usize;
                    check(
                        raw::mlx_device_info_get_size(&mut value, info, key_c.as_ptr()),
                        "mlx_device_info_get_size",
                    )?;
                    println!("  {key}: {value}");
                }
            }
            let _ = raw::mlx_vector_string_free(keys);
        }

        let _ = raw::mlx_device_info_free(info);
        let _ = raw::mlx_device_free(dev);
        println!("==================================================");
        Ok(())
    }
}

fn main() -> Result<()> {
    unsafe {
        let mut version = raw::mlx_string_new();
        check(raw::mlx_version(&mut version), "mlx_version")?;
        println!("MLX version: {}", support::mlx_string_to_rust(version));
        let _ = raw::mlx_string_free(version);

        gpu_info()?;

        let stream = raw::mlx_default_cpu_stream_new();
        let data = [1f32, 2., 3., 4., 5., 6.];
        let shape = [2, 3];
        let mut arr = raw::mlx_array_new_data(
            data.as_ptr().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            raw::mlx_dtype__MLX_FLOAT32,
        );
        print_array("hello world!", arr)?;

        let two = raw::mlx_array_new_int(2);
        check(raw::mlx_divide(&mut arr, arr, two, stream), "mlx_divide")?;
        print_array("divide by 2!", arr)?;

        check(
            raw::mlx_arange(&mut arr, 0.0, 3.0, 0.5, raw::mlx_dtype__MLX_FLOAT32, stream),
            "mlx_arange",
        )?;
        print_array("arange", arr)?;

        let mut managed = Box::new([0f32, 1., 2., 3., 4., 5.]);
        let data_ptr = managed.as_mut_ptr();
        let payload = Box::into_raw(managed);
        let arr_managed = raw::mlx_array_new_data_managed_payload(
            data_ptr.cast(),
            shape.as_ptr(),
            shape.len() as i32,
            raw::mlx_dtype__MLX_FLOAT32,
            payload.cast(),
            Some(support::drop_boxed_f32x6),
        );
        print_array("from user buffer", arr_managed)?;

        let _ = raw::mlx_array_free(arr);
        let _ = raw::mlx_array_free(two);
        let _ = raw::mlx_array_free(arr_managed);
        let _ = raw::mlx_stream_free(stream);
    }
    Ok(())
}
