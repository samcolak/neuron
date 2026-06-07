fn main() {
    if let Err(err) = apple_mlx::demo_complex_matmul() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
