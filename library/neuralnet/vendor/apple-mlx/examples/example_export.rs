#![allow(unsafe_op_in_unsafe_fn)]

#[path = "support/mod.rs"]
mod support;

use apple_mlx::raw;
use std::env;
use support::{Result, check, cstring, print_array, vector_array_get};

unsafe extern "C" fn inc_fun(res: *mut raw::mlx_array, args: raw::mlx_array) -> i32 {
    let stream = support::preferred_stream();
    let value = raw::mlx_array_new_float(1.0);
    let status = raw::mlx_add(res, args, value, stream);
    let _ = raw::mlx_stream_free(stream);
    let _ = raw::mlx_array_free(value);
    status
}

unsafe extern "C" fn mul_fun(
    out: *mut raw::mlx_vector_array,
    _args: raw::mlx_vector_array,
    kwargs: raw::mlx_map_string_to_array,
) -> i32 {
    let stream = support::preferred_stream();
    let mut x = raw::mlx_array_new();
    let mut y = raw::mlx_array_new();
    let mut res = raw::mlx_array_new();
    let x_key = c"x";
    let y_key = c"y";
    let _ = raw::mlx_map_string_to_array_get(&mut x, kwargs, x_key.as_ptr());
    let _ = raw::mlx_map_string_to_array_get(&mut y, kwargs, y_key.as_ptr());
    let _ = raw::mlx_multiply(&mut res, x, y, stream);
    let _ = raw::mlx_vector_array_set_value(out, res);
    let _ = raw::mlx_array_free(res);
    let _ = raw::mlx_array_free(y);
    let _ = raw::mlx_array_free(x);
    let _ = raw::mlx_stream_free(stream);
    0
}

fn main() -> Result<()> {
    unsafe {
        let mut x = raw::mlx_array_new_float(1.0);
        print_array("x: ", x)?;

        let inc_path = env::temp_dir().join("inc_func.bin");
        let mul_path = env::temp_dir().join("mul_func.bin");
        let inc_path_c = cstring(inc_path.to_string_lossy().as_ref())?;
        let mul_path_c = cstring(mul_path.to_string_lossy().as_ref())?;

        println!("storing inc() function in {}", inc_path.display());
        let args = raw::mlx_vector_array_new_value(x);
        let cls = raw::mlx_closure_new_unary(Some(inc_fun));
        check(
            raw::mlx_export_function(inc_path_c.as_ptr(), cls, args, false),
            "mlx_export_function",
        )?;
        let _ = raw::mlx_closure_free(cls);

        println!("loading inc() function from {}", inc_path.display());
        let xfunc_inc = raw::mlx_imported_function_new(inc_path_c.as_ptr());

        println!("evaluating inc() over x");
        let mut res = raw::mlx_vector_array_new();
        check(
            raw::mlx_imported_function_apply(&mut res, xfunc_inc, args),
            "mlx_imported_function_apply",
        )?;
        let mut y = vector_array_get(res, 0)?;
        print_array("+1: ", y)?;
        check(raw::mlx_array_set(&mut x, y), "mlx_array_set x")?;

        println!("evaluating inc() over x with kwargs");
        let empty_args = raw::mlx_vector_array_new();
        let kwargs = raw::mlx_map_string_to_array_new();
        let x_key = cstring("x")?;
        check(
            raw::mlx_map_string_to_array_insert(kwargs, x_key.as_ptr(), x),
            "mlx_map_string_to_array_insert",
        )?;
        check(
            raw::mlx_imported_function_apply_kwargs(&mut res, xfunc_inc, empty_args, kwargs),
            "mlx_imported_function_apply_kwargs",
        )?;
        let _ = raw::mlx_array_free(y);
        y = vector_array_get(res, 0)?;
        print_array("+1: ", y)?;
        check(raw::mlx_array_set(&mut x, y), "mlx_array_set x")?;

        println!("storing mul() function in {}", mul_path.display());
        let y_key = cstring("y")?;
        check(
            raw::mlx_map_string_to_array_insert(kwargs, y_key.as_ptr(), x),
            "mlx_map_string_to_array_insert y",
        )?;
        let cls_kwargs = raw::mlx_closure_kwargs_new_func(Some(mul_fun));
        check(
            raw::mlx_export_function_kwargs(
                mul_path_c.as_ptr(),
                cls_kwargs,
                empty_args,
                kwargs,
                false,
            ),
            "mlx_export_function_kwargs",
        )?;
        let _ = raw::mlx_closure_kwargs_free(cls_kwargs);

        println!("loading mul() function from {}", mul_path.display());
        let xfunc_mul = raw::mlx_imported_function_new(mul_path_c.as_ptr());
        println!("evaluating mul() over x and x with kwargs");
        print_array("x: ", x)?;
        check(
            raw::mlx_map_string_to_array_insert(kwargs, x_key.as_ptr(), x),
            "mlx_map_string_to_array_insert x",
        )?;
        check(
            raw::mlx_map_string_to_array_insert(kwargs, y_key.as_ptr(), x),
            "mlx_map_string_to_array_insert y",
        )?;
        check(
            raw::mlx_imported_function_apply_kwargs(&mut res, xfunc_mul, empty_args, kwargs),
            "mlx_imported_function_apply_kwargs mul",
        )?;
        let _ = raw::mlx_array_free(y);
        y = vector_array_get(res, 0)?;
        print_array("3*3: ", y)?;

        let _ = raw::mlx_array_free(y);
        let _ = raw::mlx_vector_array_free(res);
        let _ = raw::mlx_map_string_to_array_free(kwargs);
        let _ = raw::mlx_vector_array_free(args);
        let _ = raw::mlx_vector_array_free(empty_args);
        let _ = raw::mlx_array_free(x);
        let _ = raw::mlx_imported_function_free(xfunc_inc);
        let _ = raw::mlx_imported_function_free(xfunc_mul);
    }
    Ok(())
}
