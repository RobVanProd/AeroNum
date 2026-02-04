# AeroNum — Minimal ndarray/ufunc API target (v0)

Goal: define the smallest, testable surface that makes AeroNum feel real to users and stays friendly to Python bindings.

## v0 scope
- CPU only
- Start with **float32** (add int32 later)
- Contiguous buffers, with support for **non-contiguous views** via (shape, strides, offset)

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
- `reshape(new_shape)`: only for contiguous arrays
- Internal helper: `view(shape, strides, offset)`

## Broadcasting (v0)
NumPy-like broadcasting for elementwise operations:
- align trailing dimensions
- dims compatible if equal or 1
- output dim is max along each axis

## Definition of done (tests)
- shape/stride invariants
- broadcasting matrix:
  - (3,1) + (1,4) -> (3,4)
  - (4,) + (3,4) -> (3,4)
- matmul:
  - (m,k) @ (k,n) -> (m,n)
- sum:
  - axis=None returns scalar
  - axis=0 and axis=1 produce correct shapes

## Python binding demo (v0)
Expose an `ndarray` type with:
- `.shape`, `.dtype`
- `__add__`, `__matmul__` (or `matmul`), `sum`

Keep v0 brutally small; add features only when tests + bindings demand it.
