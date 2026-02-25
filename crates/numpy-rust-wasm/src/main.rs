use std::process::ExitCode;

fn main() -> ExitCode {
    rustpython::run(|vm| {
        numpy_rust_python::add_numpy_module(vm);
    })
}
