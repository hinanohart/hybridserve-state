//! PyO3 bindings exposing the `hybridserve_state_core` `.hss` container
//! reader/writer to Python as the `hybridserve_state._core` extension module.
//!
//! These bindings are a thin marshaling layer only: all validation and the
//! untrusted-input boundary live in the `core` crate. Tensor bytes cross the
//! boundary as Python `bytes`; the Python `io` layer wraps them in NumPy arrays.

use std::collections::{BTreeMap, HashMap};
use std::str::FromStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use hybridserve_state_core as hss;

/// `(metadata, [(name, dtype, shape, data_bytes), ...])`.
type ReadResult = (
    HashMap<String, String>,
    Vec<(String, String, Vec<u64>, Py<PyBytes>)>,
);
/// `(metadata, [(name, dtype, shape, nbytes), ...])`.
type InspectResult = (
    HashMap<String, String>,
    Vec<(String, String, Vec<u64>, usize)>,
);

fn to_py_err(e: hss::HssError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Serialize tensors to a `.hss` file.
///
/// `names`, `dtypes`, `shapes`, and `buffers` are parallel lists of equal
/// length. `metadata` is a string->string map written under `__metadata__`.
#[pyfunction]
fn write(
    path: &str,
    names: Vec<String>,
    dtypes: Vec<String>,
    shapes: Vec<Vec<u64>>,
    buffers: Vec<Vec<u8>>,
    metadata: HashMap<String, String>,
) -> PyResult<()> {
    if !(names.len() == dtypes.len() && names.len() == shapes.len() && names.len() == buffers.len())
    {
        return Err(PyValueError::new_err(
            "names, dtypes, shapes, and buffers must have equal length",
        ));
    }
    let meta: BTreeMap<String, String> = metadata.into_iter().collect();
    let mut tensors = Vec::with_capacity(names.len());
    for (((name, dtype), shape), data) in names.into_iter().zip(dtypes).zip(shapes).zip(buffers) {
        let dt = hss::Dtype::from_str(&dtype).map_err(to_py_err)?;
        tensors.push(hss::OwnedTensor {
            name,
            dtype: dt,
            shape,
            data,
        });
    }
    let bytes = hss::serialize(&tensors, &meta).map_err(to_py_err)?;
    std::fs::write(path, bytes)
        .map_err(|e| PyValueError::new_err(format!("failed to write {path}: {e}")))?;
    Ok(())
}

/// Read a `.hss` file fully.
///
/// Returns `(metadata, [(name, dtype, shape, data_bytes), ...])`.
#[pyfunction]
fn read(py: Python<'_>, path: &str) -> PyResult<ReadResult> {
    let buf = std::fs::read(path)
        .map_err(|e| PyValueError::new_err(format!("failed to read {path}: {e}")))?;
    let view = hss::parse(&buf).map_err(to_py_err)?;

    let metadata: HashMap<String, String> = view
        .metadata()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let mut tensors = Vec::new();
    for (name, info) in view.iter() {
        // Safe: `name` comes from the view's own tensor map, offsets validated.
        let bytes = view
            .tensor_bytes(name)
            .expect("tensor present in view must have bytes");
        let py_bytes = PyBytes::new(py, bytes).unbind();
        tensors.push((
            name.clone(),
            info.dtype.as_str().to_string(),
            info.shape.clone(),
            py_bytes,
        ));
    }
    Ok((metadata, tensors))
}

/// Inspect a `.hss` file's structure without copying tensor data into Python.
///
/// Returns `(metadata, [(name, dtype, shape, nbytes), ...])`.
#[pyfunction]
fn inspect(path: &str) -> PyResult<InspectResult> {
    let buf = std::fs::read(path)
        .map_err(|e| PyValueError::new_err(format!("failed to read {path}: {e}")))?;
    let view = hss::parse(&buf).map_err(to_py_err)?;

    let metadata: HashMap<String, String> = view
        .metadata()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let mut tensors = Vec::new();
    for (name, info) in view.iter() {
        let nbytes = info.data_offsets[1] - info.data_offsets[0];
        tensors.push((
            name.clone(),
            info.dtype.as_str().to_string(),
            info.shape.clone(),
            nbytes,
        ));
    }
    Ok((metadata, tensors))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__hss_version__", hss::HSS_VERSION)?;
    m.add_function(wrap_pyfunction!(write, m)?)?;
    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(inspect, m)?)?;
    Ok(())
}
