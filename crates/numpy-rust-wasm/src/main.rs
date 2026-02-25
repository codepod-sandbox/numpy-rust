use std::process::ExitCode;

use rustpython::{InterpreterBuilder, InterpreterBuilderExt};

pub fn main() -> ExitCode {
    // Add the python/ directory to sys.path so `import numpy` finds our
    // Python package (which wraps the _numpy_native Rust module).
    let python_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent()?.parent()?.parent().map(|p| p.join("python")))
        .unwrap_or_else(|| std::path::PathBuf::from("python"));

    std::env::set_var(
        "PYTHONPATH",
        match std::env::var("PYTHONPATH") {
            Ok(existing) => format!("{}:{}", python_dir.display(), existing),
            Err(_) => python_dir.display().to_string(),
        },
    );

    let config = InterpreterBuilder::new().init_stdlib();
    let numpy_def = numpy_rust_python::numpy_module_def(&config.ctx);
    let config = config.add_native_module(numpy_def);
    rustpython::run(config)
}
