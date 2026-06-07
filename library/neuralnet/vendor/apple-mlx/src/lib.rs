//! Rust bindings for Apple MLX through the official `mlx-c` C API.
//!
//! `raw` exposes the generated low-level bindings.
//! The top-level types provide a small safe wrapper over a subset of the API.

use std::error::Error as StdError;
use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::c_int;
use std::ptr;
use std::slice;

pub mod raw {
    #![allow(
        clippy::all,
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        unsafe_op_in_unsafe_fn
    )]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

type MlxArrayRaw = raw::mlx_array;
type MlxDeviceRaw = raw::mlx_device;
type MlxStreamRaw = raw::mlx_stream;
type MlxDeviceInfoRaw = raw::mlx_device_info;

const MLX_DTYPE_COMPLEX64: raw::mlx_dtype = raw::mlx_dtype__MLX_COMPLEX64;
const MLX_DEVICE_CPU: raw::mlx_device_type = raw::mlx_device_type__MLX_CPU;
const MLX_DEVICE_GPU: raw::mlx_device_type = raw::mlx_device_type__MLX_GPU;

#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl StdError for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn check(code: c_int, context: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error(format!(
            "{context} failed with MLX error code {code}"
        )))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
}

impl fmt::Display for Complex32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3}{:+.3}i", self.re, self.im)
    }
}

struct DeviceInfo {
    raw: MlxDeviceInfoRaw,
}

impl DeviceInfo {
    fn load(device: &Device) -> Result<Self> {
        let mut raw = MlxDeviceInfoRaw {
            ctx: ptr::null_mut(),
        };
        unsafe {
            check(
                raw::mlx_device_info_get(&mut raw, device.raw),
                "mlx_device_info_get",
            )?;
        }
        if raw.ctx.is_null() {
            return Err(Error("mlx_device_info_get returned a null handle".into()));
        }
        Ok(Self { raw })
    }

    fn get_string(&self, key: &str) -> Result<Option<String>> {
        let key = CString::new(key)
            .map_err(|_| Error(format!("device info key contains interior null: {key:?}")))?;
        let mut exists = false;
        unsafe {
            check(
                raw::mlx_device_info_has_key(&mut exists, self.raw, key.as_ptr()),
                "mlx_device_info_has_key",
            )?;
        }
        if !exists {
            return Ok(None);
        }

        let mut value = ptr::null();
        unsafe {
            check(
                raw::mlx_device_info_get_string(&mut value, self.raw, key.as_ptr()),
                "mlx_device_info_get_string",
            )?;
            if value.is_null() {
                return Ok(None);
            }
            Ok(Some(CStr::from_ptr(value).to_string_lossy().into_owned()))
        }
    }
}

impl Drop for DeviceInfo {
    fn drop(&mut self) {
        unsafe {
            let _ = raw::mlx_device_info_free(self.raw);
        }
    }
}

pub struct Device {
    raw: MlxDeviceRaw,
}

impl Device {
    pub fn gpu_if_available() -> Result<Option<Self>> {
        let raw = unsafe { raw::mlx_device_new_type(MLX_DEVICE_GPU, 0) };
        let device = Self { raw };
        let mut available = false;
        unsafe {
            check(
                raw::mlx_device_is_available(&mut available, device.raw),
                "mlx_device_is_available",
            )?;
        }
        if available {
            Ok(Some(device))
        } else {
            Ok(None)
        }
    }

    pub fn cpu() -> Self {
        let raw = unsafe { raw::mlx_device_new_type(MLX_DEVICE_CPU, 0) };
        Self { raw }
    }

    pub fn preferred() -> Result<Self> {
        if let Some(gpu) = Self::gpu_if_available()? {
            return Ok(gpu);
        }
        Ok(Self::cpu())
    }

    pub fn kind(&self) -> Result<&'static str> {
        let mut kind = MLX_DEVICE_CPU;
        unsafe {
            check(
                raw::mlx_device_get_type(&mut kind, self.raw),
                "mlx_device_get_type",
            )?;
        }
        Ok(match kind {
            MLX_DEVICE_CPU => "CPU",
            MLX_DEVICE_GPU => "GPU",
            _ => "Unknown",
        })
    }

    pub fn index(&self) -> Result<i32> {
        let mut index = 0;
        unsafe {
            check(
                raw::mlx_device_get_index(&mut index, self.raw),
                "mlx_device_get_index",
            )?;
        }
        Ok(index)
    }

    pub fn name(&self) -> Result<String> {
        let info = DeviceInfo::load(self)?;
        if let Some(name) = info.get_string("device_name")? {
            return Ok(name);
        }
        Ok(format!("{} device {}", self.kind()?, self.index()?))
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = raw::mlx_device_free(self.raw);
        }
    }
}

pub struct Stream {
    raw: MlxStreamRaw,
}

impl Stream {
    pub fn new(device: &Device) -> Self {
        let raw = unsafe { raw::mlx_stream_new_device(device.raw) };
        Self { raw }
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { check(raw::mlx_synchronize(self.raw), "mlx_synchronize") }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            let _ = raw::mlx_stream_free(self.raw);
        }
    }
}

pub struct Array {
    raw: MlxArrayRaw,
}

impl Array {
    pub fn from_complex_matrix(rows: usize, cols: usize, values: &[Complex32]) -> Result<Self> {
        if rows * cols != values.len() {
            return Err(Error(format!(
                "shape {rows}x{cols} does not match {} values",
                values.len()
            )));
        }

        let shape = [rows as c_int, cols as c_int];
        let raw = unsafe {
            raw::mlx_array_new_data(
                values.as_ptr().cast(),
                shape.as_ptr(),
                shape.len() as c_int,
                MLX_DTYPE_COMPLEX64,
            )
        };

        if raw.ctx.is_null() {
            return Err(Error("mlx_array_new_data returned a null handle".into()));
        }

        Ok(Self { raw })
    }

    pub fn matmul(&self, rhs: &Self, stream: &Stream) -> Result<Self> {
        let mut out = MlxArrayRaw {
            ctx: ptr::null_mut(),
        };
        unsafe {
            check(
                raw::mlx_matmul(&mut out, self.raw, rhs.raw, stream.raw),
                "mlx_matmul",
            )?;
        }
        Ok(Self { raw: out })
    }

    pub fn max_abs_error(&self, rhs: &Self, stream: &Stream) -> Result<f32> {
        let mut delta = MlxArrayRaw {
            ctx: ptr::null_mut(),
        };
        let mut magnitude = MlxArrayRaw {
            ctx: ptr::null_mut(),
        };
        let mut max_value = MlxArrayRaw {
            ctx: ptr::null_mut(),
        };

        unsafe {
            check(
                raw::mlx_subtract(&mut delta, self.raw, rhs.raw, stream.raw),
                "mlx_subtract",
            )?;
            check(raw::mlx_abs(&mut magnitude, delta, stream.raw), "mlx_abs")?;
            check(
                raw::mlx_max(&mut max_value, magnitude, false, stream.raw),
                "mlx_max",
            )?;
            check(raw::mlx_array_eval(max_value), "mlx_array_eval")?;
            stream.synchronize()?;
            let mut value = 0.0;
            check(
                raw::mlx_array_item_float32(&mut value, max_value),
                "mlx_array_item_float32",
            )?;
            let _ = raw::mlx_array_free(delta);
            let _ = raw::mlx_array_free(magnitude);
            let _ = raw::mlx_array_free(max_value);
            Ok(value)
        }
    }

    pub fn shape(&self) -> Result<Vec<usize>> {
        unsafe {
            let ndim = raw::mlx_array_ndim(self.raw);
            let shape_ptr = raw::mlx_array_shape(self.raw);
            if shape_ptr.is_null() {
                return Err(Error("mlx_array_shape returned a null pointer".into()));
            }
            Ok(slice::from_raw_parts(shape_ptr, ndim)
                .iter()
                .map(|dim| *dim as usize)
                .collect())
        }
    }

    pub fn to_complex_vec(&self, stream: &Stream) -> Result<Vec<Complex32>> {
        unsafe {
            if raw::mlx_array_dtype(self.raw) != MLX_DTYPE_COMPLEX64 {
                return Err(Error("expected MLX complex64 output".into()));
            }
            check(raw::mlx_array_eval(self.raw), "mlx_array_eval")?;
            stream.synchronize()?;
            let count = raw::mlx_array_size(self.raw);
            let ptr = raw::mlx_array_data_complex64(self.raw) as *const Complex32;
            if ptr.is_null() {
                return Err(Error("mlx_array_data_complex64 returned null".into()));
            }
            Ok(slice::from_raw_parts(ptr, count).to_vec())
        }
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        unsafe {
            let _ = raw::mlx_array_free(self.raw);
        }
    }
}

pub fn cpu_complex_matmul(
    lhs: &[Complex32],
    rhs: &[Complex32],
    lhs_rows: usize,
    lhs_cols: usize,
    rhs_cols: usize,
) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); lhs_rows * rhs_cols];
    for row in 0..lhs_rows {
        for col in 0..rhs_cols {
            let mut acc = Complex32::new(0.0, 0.0);
            for k in 0..lhs_cols {
                let a = lhs[row * lhs_cols + k];
                let b = rhs[k * rhs_cols + col];
                acc.re += a.re * b.re - a.im * b.im;
                acc.im += a.re * b.im + a.im * b.re;
            }
            out[row * rhs_cols + col] = acc;
        }
    }
    out
}

pub fn print_matrix(values: &[Complex32], rows: usize, cols: usize, label: &str) {
    println!("{label}:");
    for row in values.chunks(cols).take(rows) {
        let rendered = row
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("  ");
        println!("  {rendered}");
    }
}

pub fn demo_complex_matmul() -> Result<()> {
    let lhs = vec![
        Complex32::new(1.0, 2.0),
        Complex32::new(3.0, -1.0),
        Complex32::new(-2.0, 0.5),
        Complex32::new(0.0, 4.0),
    ];
    let rhs = vec![
        Complex32::new(0.5, -1.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(-3.0, 1.5),
        Complex32::new(1.0, -2.0),
    ];

    let device = Device::preferred()?;
    let stream = Stream::new(&device);
    let lhs_array = Array::from_complex_matrix(2, 2, &lhs)?;
    let rhs_array = Array::from_complex_matrix(2, 2, &rhs)?;
    let product = lhs_array.matmul(&rhs_array, &stream)?;
    let product_shape = product.shape()?;
    let product_values = product.to_complex_vec(&stream)?;

    let expected_values = cpu_complex_matmul(&lhs, &rhs, 2, 2, 2);
    let expected = Array::from_complex_matrix(2, 2, &expected_values)?;
    let max_abs_error = product.max_abs_error(&expected, &stream)?;

    println!(
        "Using Apple MLX on {} device {} ({})",
        device.kind()?,
        device.index()?,
        device.name()?
    );
    println!("Output shape: {:?}", product_shape);
    print_matrix(&lhs, 2, 2, "Left matrix");
    print_matrix(&rhs, 2, 2, "Right matrix");
    print_matrix(&product_values, 2, 2, "MLX product");
    println!("Max absolute error vs CPU reference: {max_abs_error:.6}");

    if max_abs_error > 1e-4 {
        return Err(Error(format!(
            "MLX result drifted from the CPU reference: {max_abs_error}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_reference_matches_known_values() {
        let lhs = vec![
            Complex32::new(1.0, 2.0),
            Complex32::new(3.0, -1.0),
            Complex32::new(-2.0, 0.5),
            Complex32::new(0.0, 4.0),
        ];
        let rhs = vec![
            Complex32::new(0.5, -1.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(-3.0, 1.5),
            Complex32::new(1.0, -2.0),
        ];

        let actual = cpu_complex_matmul(&lhs, &rhs, 2, 2, 2);
        let expected = vec![
            Complex32::new(-5.0, 7.5),
            Complex32::new(3.0, -3.0),
            Complex32::new(-6.5, -9.75),
            Complex32::new(4.0, 5.0),
        ];

        assert_eq!(actual, expected);
    }
}
