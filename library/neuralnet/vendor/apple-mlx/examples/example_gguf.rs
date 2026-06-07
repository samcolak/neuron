#![allow(unsafe_op_in_unsafe_fn)]

#[path = "support/mod.rs"]
mod support;

use apple_mlx::raw;
use support::{Result, check, cstring, vector_string_get};

const DTYPE_STRS: [&str; 14] = [
    "MLX_BOOL",
    "MLX_UINT8",
    "MLX_UINT16",
    "MLX_UINT32",
    "MLX_UINT64",
    "MLX_INT8",
    "MLX_INT16",
    "MLX_INT32",
    "MLX_INT64",
    "MLX_FLOAT16",
    "MLX_FLOAT32",
    "MLX_FLOAT64",
    "MLX_BFLOAT16",
    "MLX_COMPLEX64",
];

unsafe fn add_zero_array_to_gguf(
    gguf: raw::mlx_io_gguf,
    key: &str,
    shape: &[i32],
    dtype: raw::mlx_dtype,
    metadata: Option<&str>,
    stream: raw::mlx_stream,
) -> Result<()> {
    let mut arr = raw::mlx_array_new();
    check(
        raw::mlx_zeros(&mut arr, shape.as_ptr(), shape.len(), dtype, stream),
        "mlx_zeros",
    )?;
    let key_c = cstring(key)?;
    check(
        raw::mlx_io_gguf_set_array(gguf, key_c.as_ptr(), arr),
        "mlx_io_gguf_set_array",
    )?;
    if let Some(metadata) = metadata {
        let metadata_c = cstring(metadata)?;
        check(
            raw::mlx_io_gguf_set_metadata_string(gguf, key_c.as_ptr(), metadata_c.as_ptr()),
            "mlx_io_gguf_set_metadata_string",
        )?;
    }
    let _ = raw::mlx_array_free(arr);
    Ok(())
}

fn main() -> Result<()> {
    unsafe {
        let stream = raw::mlx_default_cpu_stream_new();
        let filename = std::env::args()
            .nth(1)
            .unwrap_or_else(|| std::env::temp_dir().join("test.gguf").display().to_string());
        let filename_c = cstring(&filename)?;

        if std::env::args().nth(1).is_none() {
            let gguf = raw::mlx_io_gguf_new();
            add_zero_array_to_gguf(
                gguf,
                "array3D",
                &[3, 4, 5],
                raw::mlx_dtype__MLX_FLOAT16,
                None,
                stream,
            )?;
            add_zero_array_to_gguf(
                gguf,
                "array2D",
                &[6, 7],
                raw::mlx_dtype__MLX_FLOAT32,
                Some("a 6x7 zero array"),
                stream,
            )?;
            check(
                raw::mlx_save_gguf(filename_c.as_ptr(), gguf),
                "mlx_save_gguf",
            )?;
            let _ = raw::mlx_io_gguf_free(gguf);
        }

        let mut gguf = raw::mlx_io_gguf_new();
        check(
            raw::mlx_load_gguf(&mut gguf, filename_c.as_ptr(), stream),
            "mlx_load_gguf",
        )?;
        let mut keys = raw::mlx_vector_string_new();
        check(
            raw::mlx_io_gguf_get_keys(&mut keys, gguf),
            "mlx_io_gguf_get_keys",
        )?;

        let mut value = raw::mlx_array_new();
        for i in 0..raw::mlx_vector_string_size(keys) {
            let key = vector_string_get(keys, i)?;
            let key_c = cstring(&key)?;
            print!("{key} ");
            check(
                raw::mlx_io_gguf_get_array(&mut value, gguf, key_c.as_ptr()),
                "mlx_io_gguf_get_array",
            )?;
            let shape = raw::mlx_array_shape(value);
            for d in 0..raw::mlx_array_ndim(value) {
                if d != 0 {
                    print!("x");
                }
                print!("{}", *shape.add(d));
            }
            let dtype = raw::mlx_array_dtype(value) as usize;
            print!(" {}", DTYPE_STRS[dtype]);

            let mut flag = false;
            let _ = raw::mlx_io_gguf_has_metadata_array(&mut flag, gguf, key_c.as_ptr());
            if flag {
                print!(" [array]");
            }
            let _ = raw::mlx_io_gguf_has_metadata_string(&mut flag, gguf, key_c.as_ptr());
            if flag {
                print!(" [string]");
            }
            let _ = raw::mlx_io_gguf_has_metadata_vector_string(&mut flag, gguf, key_c.as_ptr());
            if flag {
                print!(" [vector string]");
            }
            println!();
        }

        let _ = raw::mlx_array_free(value);
        let _ = raw::mlx_vector_string_free(keys);
        let _ = raw::mlx_io_gguf_free(gguf);
        let _ = raw::mlx_stream_free(stream);
    }
    Ok(())
}
