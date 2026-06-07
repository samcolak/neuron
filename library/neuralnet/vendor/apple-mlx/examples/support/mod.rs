#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use apple_mlx::raw;
use std::error::Error as StdError;
use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::c_void;
use std::path::PathBuf;
use std::ptr;

#[derive(Debug)]
pub struct ExampleError(pub String);

impl fmt::Display for ExampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl StdError for ExampleError {}

pub type Result<T> = std::result::Result<T, ExampleError>;

pub fn check(code: i32, context: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(ExampleError(format!(
            "{context} failed with MLX error code {code}"
        )))
    }
}

pub fn cstring(value: &str) -> Result<CString> {
    CString::new(value).map_err(|_| ExampleError(format!("invalid CString: {value:?}")))
}

pub unsafe fn mlx_string_to_rust(str_: raw::mlx_string) -> String {
    let data = raw::mlx_string_data(str_);
    if data.is_null() {
        String::new()
    } else {
        CStr::from_ptr(data).to_string_lossy().into_owned()
    }
}

pub unsafe fn array_to_string(arr: raw::mlx_array) -> Result<String> {
    let mut str_ = raw::mlx_string_new();
    check(
        raw::mlx_array_tostring(&mut str_, arr),
        "mlx_array_tostring",
    )?;
    let rendered = mlx_string_to_rust(str_);
    let _ = raw::mlx_string_free(str_);
    Ok(rendered)
}

pub unsafe fn print_array(label: &str, arr: raw::mlx_array) -> Result<()> {
    println!("{label}");
    println!("{}", array_to_string(arr)?);
    Ok(())
}

pub unsafe fn vector_array_get(vec: raw::mlx_vector_array, index: usize) -> Result<raw::mlx_array> {
    let mut arr = raw::mlx_array_new();
    check(
        raw::mlx_vector_array_get(&mut arr, vec, index),
        "mlx_vector_array_get",
    )?;
    Ok(arr)
}

pub unsafe fn vector_string_get(vec: raw::mlx_vector_string, index: usize) -> Result<String> {
    let mut data = ptr::null_mut();
    check(
        raw::mlx_vector_string_get(&mut data, vec, index),
        "mlx_vector_string_get",
    )?;
    if data.is_null() {
        return Ok(String::new());
    }
    Ok(CStr::from_ptr(data).to_string_lossy().into_owned())
}

pub fn vendor_example_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("vendor")
        .join("mlx-c")
        .join("examples")
        .join(name)
}

pub unsafe fn preferred_stream() -> raw::mlx_stream {
    let mut gpu_count = 0;
    if raw::mlx_device_count(&mut gpu_count, raw::mlx_device_type__MLX_GPU) == 0 && gpu_count > 0 {
        raw::mlx_default_gpu_stream_new()
    } else {
        raw::mlx_default_cpu_stream_new()
    }
}

pub unsafe extern "C" fn drop_boxed_f32x6(payload: *mut c_void) {
    if !payload.is_null() {
        drop(Box::from_raw(payload as *mut [f32; 6]));
    }
}

pub unsafe extern "C" fn noop_error_handler(msg: *const std::os::raw::c_char, _data: *mut c_void) {
    if msg.is_null() {
        println!("ignoring the error: <null>");
    } else {
        println!(
            "ignoring the error: <{}>",
            CStr::from_ptr(msg).to_string_lossy()
        );
    }
}
