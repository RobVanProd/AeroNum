# AeroNum — Minimal ndarray/ufunc API target (v0)

Goal: define the smallest, testable surface that makes AeroNum feel real to users and stays friendly to Python bindings.

## v0 scope
- CPU only
- Start with **float32** (add int32 later)
- Contiguous buffers, with support for **non-contiguous views** via (shape, strides, offset)

### dtype planning note (post-v0)
Keep the v0 implementation monomorphic (`Vec<f32>`) to keep the surface small and tests fast.
When adding more dtypes, prefer an explicit `DType` enum + a single ndarray struct holding a typed buffer
(e.g., `Vec<u8>` with reinterpretation) rather than proliferating generic `NdArray<T>` across the public API.

## ndarray object model
- `dtype`: `float32` (v0)
- `shape`: list/tuple of `int`
- `strides`: strides in *elements* (not bytes)
- `data`: contiguous allocation
- `offset`: element offset into `data` (for views)
- `device`: `cpu` only (v0)

## Constructors (v0)
- `zeros(shape)`
- `ones(shape)`
- `arange(n)` (1D)
- `from_list(list, shape?)` (optional)

## Ops (v0)
### Elementwise ufuncs
- `add`, `sub`, `mul`, `div`

### Reductions
- `sum(axis=None, keepdims=false)`

### Linalg-lite
- `matmul` (2D only in v0)

## Indexing / views (v0)
- Slicing support: **slice on first axis only**
- `reshape(new_shape)`: only for **C-contiguous** layouts (contiguous *views* are allowed even if `offset != 0`).
  - v0 can implement reshape as a copy into a new owned contiguous buffer (simplest), as long as the element order matches row-major.
- Internal helper: `view(shape, strides, offset)`

## Broadcasting (v0)
NumPy-like broadcasting for elementwise operations:
- align trailing dimensions
- dims compatible if equal or 1
- output dim is max along each axis

## Definition of done (tests)
- shape/stride invariants (including views with non-zero `offset`)
- reshape:
  - contiguous reshape preserves row-major element order
  - non-contiguous layouts (e.g. transpose-like strides) return `None`
  - zero-sized dimensions round-trip correctly
- broadcasting:
  - (3,1) + (1,4) -> (3,4)
  - (4,) + (3,4) -> (3,4)
  - scalar + matrix and a small higher-rank case to ensure trailing-dim alignment
  - incompatible shapes must error/panic
- matmul (2D only):
  - (m,k) @ (k,n) -> (m,n)
  - inner-dimension mismatch must error/panic
  - a stride-based case (e.g. transposed view) to ensure we respect strides/offset
- sum:
  - axis=None returns scalar
  - axis=0 and axis=1 produce correct shapes

## Python binding demo (v0)
Expose an `ndarray` type with:
- `.shape`, `.dtype`
- `__add__`, `__matmul__` (or `matmul`), `sum`

Keep v0 brutally small; add features only when tests + bindings demand it.
