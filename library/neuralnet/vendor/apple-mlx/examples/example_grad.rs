#![allow(unsafe_op_in_unsafe_fn)]

#[path = "support/mod.rs"]
mod support;

use apple_mlx::raw;
use std::os::raw::c_void;
use support::{Result, check, print_array, vector_array_get};

unsafe extern "C" fn inc_fun(res: *mut raw::mlx_array, input: raw::mlx_array) -> i32 {
    let stream = support::preferred_stream();
    let value = raw::mlx_array_new_float(1.0);
    let status = raw::mlx_add(res, input, value, stream);
    let _ = raw::mlx_stream_free(stream);
    let _ = raw::mlx_array_free(value);
    status
}

unsafe extern "C" fn inc_fun_value(
    out: *mut raw::mlx_vector_array,
    input: raw::mlx_vector_array,
    payload: *mut c_void,
) -> i32 {
    let stream = support::preferred_stream();
    if raw::mlx_vector_array_size(input) != 1 {
        eprintln!("inc_fun_value: expected 1 argument");
        let _ = raw::mlx_stream_free(stream);
        return 1;
    }
    let mut res = raw::mlx_array_new();
    let status = raw::mlx_vector_array_get(&mut res, input, 0);
    if status == 0 {
        let payload_value = *(payload as *mut raw::mlx_array);
        let _ = raw::mlx_add(&mut res, res, payload_value, stream);
        let _ = raw::mlx_vector_array_set_value(out, res);
    }
    let _ = raw::mlx_array_free(res);
    let _ = raw::mlx_stream_free(stream);
    status
}

unsafe extern "C" fn closure_dtor(ptr: *mut c_void) {
    if !ptr.is_null() {
        let boxed = Box::from_raw(ptr as *mut raw::mlx_array);
        let _ = raw::mlx_array_free(*boxed);
    }
}

fn main() -> Result<()> {
    unsafe {
        let x = raw::mlx_array_new_float(1.0);
        let y = raw::mlx_array_new_float(1.0);
        let cls = raw::mlx_closure_new_unary(Some(inc_fun));
        let cls_with_value = raw::mlx_closure_new_func_payload(
            Some(inc_fun_value),
            Box::into_raw(Box::new(y)).cast(),
            Some(closure_dtor),
        );

        {
            println!("jvp:");
            let one = raw::mlx_array_new_float(1.0);
            let primals = raw::mlx_vector_array_new_value(x);
            let tangents = raw::mlx_vector_array_new_value(one);
            let mut out = raw::mlx_vector_array_new();
            let mut dout = raw::mlx_vector_array_new();
            check(
                raw::mlx_jvp(&mut out, &mut dout, cls, primals, tangents),
                "mlx_jvp",
            )?;
            let out_arr = vector_array_get(out, 0)?;
            let dout_arr = vector_array_get(dout, 0)?;
            print_array("out", out_arr)?;
            print_array("dout", dout_arr)?;
            let _ = raw::mlx_array_free(dout_arr);
            let _ = raw::mlx_array_free(out_arr);
            let _ = raw::mlx_vector_array_free(dout);
            let _ = raw::mlx_vector_array_free(out);
            let _ = raw::mlx_vector_array_free(tangents);
            let _ = raw::mlx_vector_array_free(primals);
            let _ = raw::mlx_array_free(one);
        }

        {
            println!("value_and_grad:");
            let garg = [0];
            let mut vag = raw::mlx_closure_value_and_grad_new();
            check(
                raw::mlx_value_and_grad(&mut vag, cls, garg.as_ptr(), garg.len()),
                "mlx_value_and_grad",
            )?;
            let inputs = raw::mlx_vector_array_new_value(x);
            let mut out = raw::mlx_vector_array_new();
            let mut dout = raw::mlx_vector_array_new();
            check(
                raw::mlx_closure_value_and_grad_apply(&mut out, &mut dout, vag, inputs),
                "mlx_closure_value_and_grad_apply",
            )?;
            let out_arr = vector_array_get(out, 0)?;
            let dout_arr = vector_array_get(dout, 0)?;
            print_array("out", out_arr)?;
            print_array("dout", dout_arr)?;
            let _ = raw::mlx_array_free(dout_arr);
            let _ = raw::mlx_array_free(out_arr);
            let _ = raw::mlx_vector_array_free(inputs);
            let _ = raw::mlx_vector_array_free(dout);
            let _ = raw::mlx_vector_array_free(out);
            let _ = raw::mlx_closure_value_and_grad_free(vag);
        }

        {
            println!("value_and_grad with payload:");
            let garg = [0];
            let mut vag = raw::mlx_closure_value_and_grad_new();
            check(
                raw::mlx_value_and_grad(&mut vag, cls_with_value, garg.as_ptr(), garg.len()),
                "mlx_value_and_grad",
            )?;
            let inputs = raw::mlx_vector_array_new_value(x);
            let mut out = raw::mlx_vector_array_new();
            let mut dout = raw::mlx_vector_array_new();
            check(
                raw::mlx_closure_value_and_grad_apply(&mut out, &mut dout, vag, inputs),
                "mlx_closure_value_and_grad_apply",
            )?;
            let out_arr = vector_array_get(out, 0)?;
            let dout_arr = vector_array_get(dout, 0)?;
            print_array("out", out_arr)?;
            print_array("dout", dout_arr)?;
            let _ = raw::mlx_array_free(dout_arr);
            let _ = raw::mlx_array_free(out_arr);
            let _ = raw::mlx_vector_array_free(inputs);
            let _ = raw::mlx_vector_array_free(dout);
            let _ = raw::mlx_vector_array_free(out);
            let _ = raw::mlx_closure_value_and_grad_free(vag);
        }

        let _ = raw::mlx_closure_free(cls_with_value);
        let _ = raw::mlx_closure_free(cls);
        let _ = raw::mlx_array_free(x);
    }
    Ok(())
}
