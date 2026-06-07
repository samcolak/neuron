#![allow(unsafe_op_in_unsafe_fn)]

#[path = "support/mod.rs"]
mod support;

use apple_mlx::raw;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::ptr;
use support::{Result, check, noop_error_handler, print_array, vector_array_get};

#[repr(C)]
struct BogusPayload {
    value: raw::mlx_array,
    error: [u8; 256],
}

unsafe extern "C" fn inc_fun(res: *mut raw::mlx_array, input: raw::mlx_array) -> i32 {
    let stream = support::preferred_stream();
    let value = raw::mlx_array_new_float(1.0);
    let status = raw::mlx_add(res, input, value, stream);
    let _ = raw::mlx_stream_free(stream);
    let _ = raw::mlx_array_free(value);
    status
}

unsafe extern "C" fn inc_fun_bogus(
    out: *mut raw::mlx_vector_array,
    input: raw::mlx_vector_array,
    payload: *mut c_void,
) -> i32 {
    let payload = &mut *(payload as *mut BogusPayload);
    let stream = support::preferred_stream();
    if raw::mlx_vector_array_size(input) != 1 {
        eprintln!("inc_fun_bogus: expected 1 argument");
        let _ = raw::mlx_stream_free(stream);
        return 1;
    }

    let mut has_nan_flag = false;
    let mut has_nan = raw::mlx_array_new();
    let _ = raw::mlx_isnan(&mut has_nan, payload.value, stream);
    let _ = raw::mlx_any(&mut has_nan, has_nan, false, stream);
    let _ = raw::mlx_array_item_bool(&mut has_nan_flag, has_nan);
    let _ = raw::mlx_array_free(has_nan);

    if has_nan_flag {
        let _ = raw::mlx_stream_free(stream);
        payload.error.fill(0);
        payload.error[..12].copy_from_slice(b"nan detected");
        return 1;
    }

    let mut res = raw::mlx_array_new();
    let status = raw::mlx_vector_array_get(&mut res, input, 0);
    if status == 0 {
        let _ = raw::mlx_add(&mut res, res, payload.value, stream);
        let _ = raw::mlx_vector_array_set_value(out, res);
    }
    let _ = raw::mlx_array_free(res);
    let _ = raw::mlx_stream_free(stream);
    status
}

fn main() -> Result<()> {
    unsafe {
        let x = raw::mlx_array_new_float(1.0);
        print_array("x: ", x)?;

        let mut y = raw::mlx_array_new();
        let mut vy = raw::mlx_vector_array_new();
        let vx = raw::mlx_vector_array_new_value(x);
        let cls = raw::mlx_closure_new_unary(Some(inc_fun));
        check(
            raw::mlx_closure_apply(&mut vy, cls, vx),
            "mlx_closure_apply",
        )?;
        y = vector_array_get(vy, 0)?;
        print_array("+1: ", y)?;

        let mut payload = Box::new(BogusPayload {
            value: raw::mlx_array_new_float(2.0),
            error: [0; 256],
        });
        let cls_with_value = raw::mlx_closure_new_func_payload(
            Some(inc_fun_bogus),
            (&mut *payload as *mut BogusPayload).cast(),
            None,
        );
        check(
            raw::mlx_closure_apply(&mut vy, cls_with_value, vx),
            "mlx_closure_apply payload",
        )?;
        let _ = raw::mlx_array_free(y);
        y = vector_array_get(vy, 0)?;
        print_array("+2: ", y)?;

        raw::mlx_set_error_handler(Some(noop_error_handler), ptr::null_mut(), None);
        check(
            raw::mlx_array_set_float(&mut payload.value, f32::NAN),
            "mlx_array_set_float",
        )?;
        if raw::mlx_closure_apply(&mut vy, cls_with_value, vx) != 0 {
            let nul_pos = payload
                .error
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(payload.error.len());
            let rendered = String::from_utf8_lossy(&payload.error[..nul_pos]);
            println!("closure failed with: <{rendered}>");
        } else {
            let _ = raw::mlx_array_free(y);
            y = vector_array_get(vy, 0)?;
            print_array("+nan: ", y)?;
        }
        raw::mlx_set_error_handler(None, ptr::null_mut(), None);

        let _ = raw::mlx_array_free(x);
        let _ = raw::mlx_array_free(y);
        let _ = raw::mlx_array_free(payload.value);
        let _ = raw::mlx_vector_array_free(vx);
        let _ = raw::mlx_vector_array_free(vy);
        let _ = raw::mlx_closure_free(cls);
        let _ = raw::mlx_closure_free(cls_with_value);

        let _ = CStr::from_ptr(c"ok".as_ptr() as *const c_char);
    }
    Ok(())
}
