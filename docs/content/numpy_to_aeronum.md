# From NumPy to AeroNum in 10 Minutes

AeroNum provides the identical ergonomic vectorization limits standard Python researchers are accustomed to, executing entirely zero-cost over `C-level` compiled native bounds. 

## 1. Array Initialization
**NumPy (Python)**
```python
import numpy as np

arr = np.array([1.0, 2.0, 3.0])
zeros = np.zeros((3, 3))
```

**AeroNum (Aero)**
```rust
use aeronum::{Array, Array2, zeros};

let arr = Array::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
let zeroes = zeros(&[3, 3]); 
```

## 2. Mathematical Matrix Projections
**NumPy**
```python
result = np.dot(matrix_a, matrix_b)
```

**AeroNum**
```rust
let result = matrix_a.matmul(&matrix_b).unwrap(); // Explicit zero-cost binding
```

## 3. GPU Hardware Execution Constraints
**PyTorch**
```python
tensor.cuda() 
```

**AeroNum**
```rust
// Moves underlying contiguous vectors directly onto OpenCL / CUDA devices natively
tensor.to("cuda");
```

Ready to validate? **[Try it interactively in the Playground!](/playground)**
