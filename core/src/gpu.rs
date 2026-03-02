use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    Rocm,
}

#[derive(Debug)]
pub enum GpuError {
    UnsupportedPlatform,
    LibraryNotFound,
    SymbolNotFound(&'static str),
    HipCallFailed { call: &'static str, code: i32 },
    InvalidCopySize { expected: usize, actual: usize },
    NoDevice,
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => {
                write!(f, "ROCm runtime is only wired for Windows right now")
            }
            Self::LibraryNotFound => write!(f, "Could not load amdhip64.dll from PATH"),
            Self::SymbolNotFound(name) => write!(f, "Missing HIP runtime symbol: {}", name),
            Self::HipCallFailed { call, code } => {
                write!(f, "HIP call {} failed with code {}", call, code)
            }
            Self::InvalidCopySize { expected, actual } => write!(
                f,
                "Copy size mismatch: expected {} bytes but got {} bytes",
                expected, actual
            ),
            Self::NoDevice => write!(f, "No HIP device available"),
        }
    }
}

impl std::error::Error for GpuError {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HipDeviceProp {
    pub name: [u8; 256],
    pub _opaque: [u8; 4096],
}

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum HipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

pub trait Device {
    fn backend(&self) -> Backend;
    fn target_triple(&self) -> &'static str;
    fn mcpu(&self) -> &'static str;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDevice {
    pub backend: Backend,
    pub device_id: i32,
}

impl GpuDevice {
    pub fn new(backend: Backend, device_id: i32) -> Self {
        Self { backend, device_id }
    }

    pub fn auto_detect() -> Self {
        if let Ok(force_backend) = std::env::var("AERONUM_GPU_BACKEND") {
            match force_backend.trim().to_ascii_lowercase().as_str() {
                "rocm" | "amd" => return Self::new(Backend::Rocm, 0),
                "cuda" | "nvidia" => return Self::new(Backend::Cuda, 0),
                "cpu" => return Self::new(Backend::Cpu, 0),
                _ => {}
            }
        }

        if HipRuntime::new(0).is_ok() {
            return Self::new(Backend::Rocm, 0);
        }

        Self::new(Backend::Cpu, 0)
    }
}

impl Device for GpuDevice {
    fn backend(&self) -> Backend {
        self.backend
    }

    fn target_triple(&self) -> &'static str {
        match self.backend {
            Backend::Rocm => "amdgcn-amd-amdhsa",
            Backend::Cuda => "nvptx64-nvidia-cuda",
            Backend::Cpu => {
                if cfg!(target_os = "windows") {
                    "x86_64-pc-windows-msvc"
                } else if cfg!(target_os = "macos") {
                    "x86_64-apple-darwin"
                } else {
                    "x86_64-pc-linux-gnu"
                }
            }
        }
    }

    fn mcpu(&self) -> &'static str {
        match self.backend {
            Backend::Rocm => "gfx1101",
            Backend::Cuda => "sm_89",
            Backend::Cpu => "x86_64",
        }
    }
}

type HipInitFn = unsafe extern "C" fn(flags: u32) -> i32;
type HipGetDeviceCountFn = unsafe extern "C" fn(count: *mut i32) -> i32;
type HipSetDeviceFn = unsafe extern "C" fn(device: i32) -> i32;
type HipGetDevicePropertiesFn = unsafe extern "C" fn(props: *mut HipDeviceProp, device: i32) -> i32;
type HipMallocFn = unsafe extern "C" fn(ptr: *mut *mut c_void, size: usize) -> i32;
type HipFreeFn = unsafe extern "C" fn(ptr: *mut c_void) -> i32;
type HipMemcpyFn = unsafe extern "C" fn(
    dst: *mut c_void,
    src: *const c_void,
    size_bytes: usize,
    kind: HipMemcpyKind,
) -> i32;
type HipDeviceSynchronizeFn = unsafe extern "C" fn() -> i32;

type LibraryHandle = *mut c_void;

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(name: *const c_char) -> LibraryHandle;
    fn GetProcAddress(module: LibraryHandle, name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: LibraryHandle) -> i32;
}

#[cfg(not(target_os = "windows"))]
#[allow(non_snake_case)]
unsafe fn LoadLibraryA(_name: *const c_char) -> LibraryHandle {
    ptr::null_mut()
}

#[cfg(not(target_os = "windows"))]
#[allow(non_snake_case)]
unsafe fn GetProcAddress(_module: LibraryHandle, _name: *const c_char) -> *mut c_void {
    ptr::null_mut()
}

#[cfg(not(target_os = "windows"))]
#[allow(non_snake_case)]
unsafe fn FreeLibrary(_module: LibraryHandle) -> i32 {
    0
}

#[derive(Debug)]
struct HipApi {
    module: LibraryHandle,
    hip_init: HipInitFn,
    hip_get_device_count: HipGetDeviceCountFn,
    hip_set_device: HipSetDeviceFn,
    hip_get_device_properties: HipGetDevicePropertiesFn,
    hip_malloc: HipMallocFn,
    hip_free: HipFreeFn,
    hip_memcpy: HipMemcpyFn,
    hip_device_synchronize: HipDeviceSynchronizeFn,
}

impl HipApi {
    fn load() -> Result<Self, GpuError> {
        if !cfg!(target_os = "windows") {
            return Err(GpuError::UnsupportedPlatform);
        }

        let module = unsafe {
            let name = CString::new("amdhip64.dll").expect("static cstring");
            LoadLibraryA(name.as_ptr())
        };

        if module.is_null() {
            return Err(GpuError::LibraryNotFound);
        }

        macro_rules! load_symbol {
            ($symbol:literal, $ty:ty) => {{
                let sym = CString::new($symbol).expect("static cstring");
                let ptr = unsafe { GetProcAddress(module, sym.as_ptr()) };
                if ptr.is_null() {
                    unsafe {
                        FreeLibrary(module);
                    }
                    return Err(GpuError::SymbolNotFound($symbol));
                }
                unsafe { std::mem::transmute::<*mut c_void, $ty>(ptr) }
            }};
        }

        Ok(Self {
            module,
            hip_init: load_symbol!("hipInit", HipInitFn),
            hip_get_device_count: load_symbol!("hipGetDeviceCount", HipGetDeviceCountFn),
            hip_set_device: load_symbol!("hipSetDevice", HipSetDeviceFn),
            hip_get_device_properties: load_symbol!(
                "hipGetDeviceProperties",
                HipGetDevicePropertiesFn
            ),
            hip_malloc: load_symbol!("hipMalloc", HipMallocFn),
            hip_free: load_symbol!("hipFree", HipFreeFn),
            hip_memcpy: load_symbol!("hipMemcpy", HipMemcpyFn),
            hip_device_synchronize: load_symbol!("hipDeviceSynchronize", HipDeviceSynchronizeFn),
        })
    }

    fn check(&self, code: i32, call: &'static str) -> Result<(), GpuError> {
        if code == 0 {
            Ok(())
        } else {
            Err(GpuError::HipCallFailed { call, code })
        }
    }

    fn init(&self) -> Result<(), GpuError> {
        let code = unsafe { (self.hip_init)(0) };
        self.check(code, "hipInit")
    }

    fn device_count(&self) -> Result<i32, GpuError> {
        let mut count = 0;
        let code = unsafe { (self.hip_get_device_count)(&mut count as *mut i32) };
        self.check(code, "hipGetDeviceCount")?;
        Ok(count)
    }

    fn set_device(&self, device_id: i32) -> Result<(), GpuError> {
        let code = unsafe { (self.hip_set_device)(device_id) };
        self.check(code, "hipSetDevice")
    }

    fn device_name(&self, device_id: i32) -> Result<String, GpuError> {
        let mut props = MaybeUninit::<HipDeviceProp>::zeroed();
        let code = unsafe { (self.hip_get_device_properties)(props.as_mut_ptr(), device_id) };
        self.check(code, "hipGetDeviceProperties")?;
        let props = unsafe { props.assume_init() };
        let name = unsafe { CStr::from_ptr(props.name.as_ptr() as *const c_char) }
            .to_string_lossy()
            .trim()
            .to_string();
        Ok(name)
    }

    fn malloc(&self, size_bytes: usize) -> Result<*mut c_void, GpuError> {
        let mut ptr = ptr::null_mut::<c_void>();
        let code = unsafe { (self.hip_malloc)(&mut ptr as *mut *mut c_void, size_bytes) };
        self.check(code, "hipMalloc")?;
        Ok(ptr)
    }

    fn free(&self, ptr: *mut c_void) -> Result<(), GpuError> {
        let code = unsafe { (self.hip_free)(ptr) };
        self.check(code, "hipFree")
    }

    fn memcpy(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: HipMemcpyKind,
    ) -> Result<(), GpuError> {
        let code = unsafe { (self.hip_memcpy)(dst, src, size_bytes, kind) };
        self.check(code, "hipMemcpy")
    }

    fn synchronize(&self) -> Result<(), GpuError> {
        let code = unsafe { (self.hip_device_synchronize)() };
        self.check(code, "hipDeviceSynchronize")
    }
}

impl Drop for HipApi {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                let _ = FreeLibrary(self.module);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct HipRuntime {
    api: Arc<HipApi>,
    device_id: i32,
}

impl HipRuntime {
    pub fn new(device_id: i32) -> Result<Self, GpuError> {
        let api = Arc::new(HipApi::load()?);
        api.init()?;
        let count = api.device_count()?;
        if count <= 0 || device_id < 0 || device_id >= count {
            return Err(GpuError::NoDevice);
        }
        api.set_device(device_id)?;
        Ok(Self { api, device_id })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn device_name(&self) -> Result<String, GpuError> {
        self.api.device_name(self.device_id)
    }

    pub fn allocate(&self, size_bytes: usize) -> Result<HipBuffer, GpuError> {
        let ptr = self.api.malloc(size_bytes)?;
        Ok(HipBuffer {
            api: Arc::clone(&self.api),
            ptr,
            size_bytes,
        })
    }

    pub fn copy_to_device<T: Copy>(&self, host: &[T]) -> Result<HipBuffer, GpuError> {
        let size_bytes = std::mem::size_of_val(host);
        let buffer = self.allocate(size_bytes)?;
        if size_bytes > 0 {
            self.api.memcpy(
                buffer.ptr,
                host.as_ptr() as *const c_void,
                size_bytes,
                HipMemcpyKind::HostToDevice,
            )?;
        }
        Ok(buffer)
    }

    pub fn copy_to_host<T: Copy>(
        &self,
        buffer: &HipBuffer,
        host: &mut [T],
    ) -> Result<(), GpuError> {
        let expected = std::mem::size_of_val(host);
        if expected != buffer.size_bytes {
            return Err(GpuError::InvalidCopySize {
                expected: buffer.size_bytes,
                actual: expected,
            });
        }
        if expected > 0 {
            self.api.memcpy(
                host.as_mut_ptr() as *mut c_void,
                buffer.ptr as *const c_void,
                expected,
                HipMemcpyKind::DeviceToHost,
            )?;
        }
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), GpuError> {
        self.api.synchronize()
    }
}

#[derive(Debug)]
pub struct HipBuffer {
    api: Arc<HipApi>,
    ptr: *mut c_void,
    size_bytes: usize,
}

impl HipBuffer {
    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = self.api.free(self.ptr);
            self.ptr = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_profile_metadata_is_stable() {
        let dev = GpuDevice::new(Backend::Rocm, 0);
        assert_eq!(dev.target_triple(), "amdgcn-amd-amdhsa");
        assert_eq!(dev.mcpu(), "gfx1101");
    }

    #[test]
    fn runtime_creation_is_result_typed() {
        let runtime = HipRuntime::new(0);
        if let Ok(rt) = runtime {
            // If HIP is available on the machine, assert we can query device metadata.
            let name = rt.device_name().unwrap_or_default();
            assert!(!name.is_empty());
        }
    }
}
