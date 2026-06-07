#![allow(unsafe_op_in_unsafe_fn)]

#[path = "support/mod.rs"]
mod support;

use apple_mlx::raw;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::ptr;
use support::{Result, check, print_array, vendor_example_path};

#[repr(C)]
struct MemStream {
    data: *mut u8,
    pos: usize,
    size: usize,
    err: bool,
    free_data: bool,
}

const SEEK_SET_: i32 = 0;
const SEEK_CUR_: i32 = 1;
const SEEK_END_: i32 = 2;

unsafe extern "C" fn mem_is_open(desc: *mut c_void) -> bool {
    println!("ISOPEN");
    !desc.is_null()
}

unsafe extern "C" fn mem_good(desc: *mut c_void) -> bool {
    println!("GOOD");
    let mem = &*(desc as *mut MemStream);
    !mem.err
}

unsafe extern "C" fn mem_tell(desc: *mut c_void) -> usize {
    println!("TELL");
    (*(desc as *mut MemStream)).pos
}

unsafe extern "C" fn mem_seek(desc: *mut c_void, off: i64, whence: i32) {
    println!("SEEK");
    let mem = &mut *(desc as *mut MemStream);
    let new_pos = match whence {
        SEEK_SET_ => off as isize,
        SEEK_CUR_ => mem.pos as isize + off as isize,
        SEEK_END_ => mem.size as isize + off as isize,
        _ => {
            mem.err = true;
            return;
        }
    };
    if new_pos < 0 || new_pos as usize > mem.size {
        mem.err = true;
    } else {
        mem.pos = new_pos as usize;
    }
}

unsafe extern "C" fn mem_read(desc: *mut c_void, data: *mut c_char, n: usize) {
    println!("READ {n}");
    let mem = &mut *(desc as *mut MemStream);
    if mem.pos + n > mem.size {
        mem.err = true;
        return;
    }
    ptr::copy_nonoverlapping(mem.data.add(mem.pos), data.cast::<u8>(), n);
    mem.pos += n;
}

unsafe extern "C" fn mem_read_at_offset(
    desc: *mut c_void,
    data: *mut c_char,
    n: usize,
    off: usize,
) {
    println!("READ@OFFSET {n} @ {off}");
    let mem = &mut *(desc as *mut MemStream);
    if off + n > mem.size {
        mem.err = true;
        return;
    }
    ptr::copy_nonoverlapping(mem.data.add(off), data.cast::<u8>(), n);
    mem.pos = off;
}

unsafe extern "C" fn mem_write(desc: *mut c_void, data: *const c_char, n: usize) {
    println!("WRITE {n}");
    let mem = &mut *(desc as *mut MemStream);
    if mem.pos + n > mem.size {
        mem.err = true;
        return;
    }
    ptr::copy_nonoverlapping(data.cast::<u8>(), mem.data.add(mem.pos), n);
    mem.pos += n;
}

unsafe extern "C" fn mem_label(_desc: *mut c_void) -> *const c_char {
    c"<custom memory stream>".as_ptr()
}

unsafe extern "C" fn mem_free(desc: *mut c_void) {
    let mem = &mut *(desc as *mut MemStream);
    if mem.free_data {
        println!("FREE DATA");
        let _ = Vec::from_raw_parts(mem.data, mem.size, mem.size);
        mem.data = ptr::null_mut();
    }
}

fn main() -> Result<()> {
    unsafe {
        let stream = raw::mlx_default_cpu_stream_new();
        let mut data = raw::mlx_map_string_to_array_new();
        let mut metadata = raw::mlx_map_string_to_string_new();
        let path = vendor_example_path("arrays.safetensors");
        let path_c = support::cstring(path.to_string_lossy().as_ref())?;

        println!("load data from disk:");
        check(
            raw::mlx_load_safetensors(&mut data, &mut metadata, path_c.as_ptr(), stream),
            "mlx_load_safetensors",
        )?;
        let mut it = raw::mlx_map_string_to_array_iterator_new(data);
        let mut key = ptr::null();
        let mut value = raw::mlx_array_new();
        while raw::mlx_map_string_to_array_iterator_next(&mut key, &mut value, it) == 0 {
            print_array(CStr::from_ptr(key).to_str().unwrap_or("<invalid>"), value)?;
        }

        println!("attempting to write arrays in a memory stream");
        let mut backing = vec![0u8; 2048];
        let mut mem_stream = Box::new(MemStream {
            data: backing.as_mut_ptr(),
            pos: 0,
            size: backing.len(),
            err: false,
            free_data: false,
        });
        std::mem::forget(backing);
        let vtable = raw::mlx_io_vtable {
            is_open: Some(mem_is_open),
            good: Some(mem_good),
            tell: Some(mem_tell),
            seek: Some(mem_seek),
            read: Some(mem_read),
            read_at_offset: Some(mem_read_at_offset),
            write: Some(mem_write),
            label: Some(mem_label),
            free: Some(mem_free),
        };
        let writer = raw::mlx_io_writer_new((&mut *mem_stream as *mut MemStream).cast(), vtable);
        check(
            raw::mlx_save_safetensors_writer(writer, data, metadata),
            "mlx_save_safetensors_writer",
        )?;
        let _ = raw::mlx_io_writer_free(writer);

        println!(
            "position in memory stream: {} err flag: {}",
            mem_stream.pos, mem_stream.err as i32
        );
        print!("data in memory stream: ");
        for byte in std::slice::from_raw_parts(mem_stream.data, mem_stream.pos) {
            print!("{}", *byte as char);
        }
        println!();

        mem_stream.pos = 0;
        let _ = raw::mlx_map_string_to_array_free(data);
        let _ = raw::mlx_map_string_to_string_free(metadata);
        let _ = raw::mlx_map_string_to_array_iterator_free(it);

        println!("attempting to read from memory");
        mem_stream.free_data = true;
        let reader = raw::mlx_io_reader_new((&mut *mem_stream as *mut MemStream).cast(), vtable);
        data = raw::mlx_map_string_to_array_new();
        metadata = raw::mlx_map_string_to_string_new();
        check(
            raw::mlx_load_safetensors_reader(&mut data, &mut metadata, reader, stream),
            "mlx_load_safetensors_reader",
        )?;
        let _ = raw::mlx_io_reader_free(reader);

        println!("now the arrays (lazily evaluated):");
        it = raw::mlx_map_string_to_array_iterator_new(data);
        while raw::mlx_map_string_to_array_iterator_next(&mut key, &mut value, it) == 0 {
            print_array(CStr::from_ptr(key).to_str().unwrap_or("<invalid>"), value)?;
        }

        let _ = raw::mlx_array_free(value);
        let _ = raw::mlx_map_string_to_array_free(data);
        let _ = raw::mlx_map_string_to_string_free(metadata);
        let _ = raw::mlx_map_string_to_array_iterator_free(it);
        let _ = raw::mlx_stream_free(stream);
        drop(mem_stream);
    }
    Ok(())
}
