use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const VENDORED_MLX_C_DIR: &str = "vendor/mlx-c";
const LINKED_MLX_PREFIX_DIR: &str = ".linked/mlx-prefix";

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let source_dir = manifest_dir.join(VENDORED_MLX_C_DIR);
    let linked_mlx_prefix_dir = manifest_dir.join(LINKED_MLX_PREFIX_DIR);
    let docs_only =
        env::var_os("CARGO_FEATURE_DOCS_ONLY").is_some() || env::var_os("DOCS_RS").is_some();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", source_dir.display());
    println!("cargo:rerun-if-changed={}", linked_mlx_prefix_dir.display());
    println!("cargo:rerun-if-env-changed=APPLE_MLX_PREFIX");
    println!("cargo:rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo:rerun-if-env-changed=MLX_DIR");
    println!("cargo:rerun-if-env-changed=MLX_BUILD_METAL");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    generate_bindings(&source_dir, &out_dir);

    if docs_only {
        return;
    }

    let metal_enabled = has_metal_toolchain();

    if !metal_enabled {
        println!("cargo:warning=Metal toolchain not available; building MLX with CPU backend only");
    }

    let mut cfg = cmake::Config::new(&source_dir);
    cfg.profile("Release")
        .define("MLX_C_BUILD_EXAMPLES", "OFF")
        .define("MLX_C_USE_SYSTEM_MLX", "ON")
        .define("BUILD_SHARED_LIBS", "ON")
        .define("MLX_BUILD_METAL", if metal_enabled { "ON" } else { "OFF" });

    let mut system_lib_dirs = Vec::new();

    let mlx_prefix = resolve_mlx_prefix(&linked_mlx_prefix_dir).unwrap_or_else(|| {
        panic!(
            "MLX support requires an explicit external MLX prefix. Set APPLE_MLX_PREFIX, MLX_DIR, or CMAKE_PREFIX_PATH, or create a symlink at {}",
            linked_mlx_prefix_dir.display()
        )
    });

    cfg.define("CMAKE_PREFIX_PATH", mlx_prefix.to_string_lossy().as_ref());

    let mlx_dir = mlx_prefix.join("share/cmake/MLX");
    if mlx_dir.exists() {
        cfg.define("MLX_DIR", mlx_dir.to_string_lossy().as_ref());
    }

    let runtime_lib_dir = mlx_prefix.join("lib");
    system_lib_dirs.push(runtime_lib_dir.clone());

    let dst = cfg.build();
    let lib_dir = dst.join("lib");

    copy_runtime_assets(
        &runtime_lib_dir,
        &lib_dir,
        &["libmlx.dylib", "libjaccl.dylib", "mlx.metallib"],
    );

    if let Some(mlx_library_dir) = infer_mlx_library_dir(&dst) {
        system_lib_dirs.push(mlx_library_dir);
    }

    system_lib_dirs.sort();
    system_lib_dirs.dedup();

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    for dir in system_lib_dirs {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=mlxc");
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-lib=dylib=c++");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    if metal_enabled {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
    }
}

fn generate_bindings(source_dir: &Path, out_dir: &Path) {
    let header = source_dir.join("mlx/c/mlx.h");
    let include_dir = source_dir.to_string_lossy().into_owned();

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg(format!("-I{include_dir}"))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_file(".*/mlx/c/.*")
        .layout_tests(false)
        .generate_comments(false)
        .generate()
        .expect("failed to generate mlx-c bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}

fn has_metal_toolchain() -> bool {
    if let Some(value) = env::var_os("MLX_BUILD_METAL") {
        let value = value.to_string_lossy();
        if value.eq_ignore_ascii_case("on") || value == "1" || value.eq_ignore_ascii_case("true") {
            return true;
        }
        if value.eq_ignore_ascii_case("off") || value == "0" || value.eq_ignore_ascii_case("false")
        {
            return false;
        }
    }

    Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-v"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn mlx_prefix_from_mlx_dir(mlx_dir: &Path) -> Option<PathBuf> {
    let mut path = mlx_dir.to_path_buf();
    for _ in 0..3 {
        path = path.parent()?.to_path_buf();
    }
    Some(path)
}

fn resolve_mlx_prefix(linked_prefix_dir: &Path) -> Option<PathBuf> {
    if let Some(prefix) = env::var_os("APPLE_MLX_PREFIX") {
        let prefix = PathBuf::from(prefix);
        if mlx_prefix_is_valid(&prefix) {
            return Some(prefix);
        }
    }

    if let Some(mlx_dir) = env::var_os("MLX_DIR") {
        let mlx_dir = PathBuf::from(mlx_dir);
        if let Some(prefix) = mlx_prefix_from_mlx_dir(&mlx_dir) {
            if mlx_prefix_is_valid(&prefix) {
                return Some(prefix);
            }
        }
    }

    if let Some(prefixes) = env::var_os("CMAKE_PREFIX_PATH") {
        for prefix in env::split_paths(&prefixes) {
            if mlx_prefix_is_valid(&prefix) {
                return Some(prefix);
            }
        }
    }

    if mlx_prefix_is_valid(linked_prefix_dir) {
        return Some(linked_prefix_dir.to_path_buf());
    }

    None
}

fn mlx_prefix_is_valid(prefix_dir: &Path) -> bool {
    let mlx_config = prefix_dir.join("share/cmake/MLX/MLXConfig.cmake");
    let mlx_lib = prefix_dir.join("lib/libmlx.dylib");
    let jaccl_lib = prefix_dir.join("lib/libjaccl.dylib");
    let metallib = prefix_dir.join("lib/mlx.metallib");

    mlx_config.exists() && mlx_lib.exists() && jaccl_lib.exists() && metallib.exists()
}

fn copy_runtime_assets(source_dir: &Path, target_dir: &Path, assets: &[&str]) {
    fs::create_dir_all(target_dir).expect("failed to create target runtime lib dir");

    for asset in assets {
        let source = source_dir.join(asset);
        let target = target_dir.join(asset);

        if source.exists() {
            if target.exists() {
                let mut permissions = fs::metadata(&target)
                    .unwrap_or_else(|error| panic!("failed to stat {}: {}", target.display(), error))
                    .permissions();
                permissions.set_readonly(false);
                fs::set_permissions(&target, permissions).unwrap_or_else(|error| {
                    panic!("failed to update permissions on {}: {}", target.display(), error)
                });
                fs::remove_file(&target).unwrap_or_else(|error| {
                    panic!("failed to remove existing runtime dylib {}: {}", target.display(), error)
                });
            }

            fs::copy(&source, &target).unwrap_or_else(|error| {
                panic!(
                    "failed to copy runtime asset {} to {}: {}",
                    source.display(),
                    target.display(),
                    error
                )
            });
            if target.extension().and_then(|ext| ext.to_str()) == Some("dylib") {
                ad_hoc_codesign(&target);
            }
        }
    }
}

fn ad_hoc_codesign(path: &Path) {
    if !cfg!(target_os = "macos") {
        return;
    }

    let status = Command::new("/usr/bin/codesign")
        .arg("--force")
        .arg("--sign")
        .arg("-")
        .arg(path)
        .status();

    match status {
        Ok(status) if status.success() => {}
        Ok(status) => panic!("failed to ad-hoc sign {}: exit status {}", path.display(), status),
        Err(error) => panic!("failed to ad-hoc sign {}: {}", path.display(), error),
    }
}

fn infer_mlx_library_dir(dst: &Path) -> Option<PathBuf> {
    let cache = dst.join("build/CMakeCache.txt");
    let mlx_library = read_cmake_cache_value(&cache, "MLX_LIBRARY")?;
    PathBuf::from(mlx_library).parent().map(Path::to_path_buf)
}

fn read_cmake_cache_value(cache_path: &Path, key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    let contents = fs::read_to_string(cache_path).ok()?;

    contents
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .and_then(|line| line.split_once('='))
        .map(|(_, value)| value.trim().to_owned())
        .filter(|value| !value.is_empty() && !value.ends_with("-NOTFOUND"))
}
