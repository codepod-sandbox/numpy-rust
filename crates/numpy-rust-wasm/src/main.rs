use std::process::ExitCode;

use rustpython::{InterpreterBuilder, InterpreterBuilderExt};

pub fn main() -> ExitCode {
    let config = InterpreterBuilder::new().init_stdlib();
    let numpy_def = numpy_rust_python::numpy_module_def(&config.ctx);
    let config = config.add_native_module(numpy_def);
    rustpython::run(config)
}
