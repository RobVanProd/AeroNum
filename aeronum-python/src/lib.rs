use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

// This is the Rust scaffold for PyO3 bindings mirroring the Aero core

#[pyclass]
pub struct PyNdArray {
    // Internal struct linking dynamically to the Aero DLL payload
    device_id: i32, // 0 = CPU, 1 = CUDA
}

#[pymethods]
impl PyNdArray {
    #[new]
    pub fn new() -> Self {
        PyNdArray { device_id: 0 }
    }

    /// Moves the tensor to the active CUDA device
    pub fn cuda(&mut self) -> PyResult<()> {
        // Blueprint:
        // Automatically allocates zero-copy memory boundaries calling the Aero
        // `DeviceArray::to_device()` bindings.
        self.device_id = 1;
        Ok(())
    }

    /// Moves the tensor payload back to main CPU memory
    pub fn cpu(&mut self) -> PyResult<()> {
        // Blueprint:
        // Returns the Array payload calling `DeviceArray::to_host()`
        self.device_id = 0;
        Ok(())
    }

    #[getter]
    pub fn is_cuda(&self) -> PyResult<bool> {
        Ok(self.device_id == 1)
    }
}

#[pymodule]
fn aeronum(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNdArray>()?;
    Ok(())
}
