import numpy as np
import matplotlib.pyplot as plt
import quimb as qu
import quimb.tensor as qtn
import cotengra as ctg
import time 
#======================================================
#This file is used to output data files of local measurement values on a random MPS. The output file is filename = f"../local_D/vary_D_results_local_haar_unitary_L{L}D{D}.npz"
#======================================================
# =====================================================
# Tool Function 1: Construction global operator X^{⊗n}（d = 2^n）
# =====================================================
def pauli_X_tensor(n):
    """
    Construct X^{⊗n}, corresponding to physical dimension d = 2^n
    """
    X = qu.pauli('X')
    op = X
    for _ in range(n - 1):
        op = np.kron(op, X)
    return op
# =====================================================
# Tool Function 2：Pool of generating random Haar unitary
# =====================================================
def generate_haar_tensor_pool(n_pool, D, d, s0=0, seed=None, dtype=np.complex128):
    """
    Pre-generate a tensor pool containing n_pool tensors after Haar random projection. The tensor shape is (D, d, D)
    """
    rng = np.random.default_rng(seed)
    pool = []
    # n is the total dimension of the unitary matrix
    n = D * d
    for _ in range(n_pool):
        # 1. Generate a Haar random unitary matrix (using QR decomposition)
        z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
        q, r = np.linalg.qr(z)
        ph = np.diag(r)
        ph = ph / np.abs(ph)
        U = (q * ph.conj()).astype(dtype)
        # 2. Reshape and project (D, d, D, d) -> (D, D, d)
        U4 = U.reshape(D, d, D, d)
        A = U4[:, s0, :, :]
        # 3. Convert shapes to quimb standard: (left, phys, right) -> (D, d, D)
        A = np.transpose(A, (0, 2, 1))
        pool.append(A)
    return pool
# =====================================================
# Tool Function 3：Randomly draw tensors from a pre-generated pool to quickly construct periodic structure MPS
# =====================================================
def build_pbc_mps_from_pool(L, D, d, tensor_pool, seed=None):
    """
    Randomly draw tensors from a pre-generated pool to quickly construct periodic structure MPS.
    """
    rng = np.random.default_rng(seed)
    pool_size = len(tensor_pool)

    def fill_fn(shape):
        # Randomly select an index from the pool
        idx = rng.integers(0, pool_size)
        return tensor_pool[idx]

    # Build periodic structure MPS
    psi = qtn.MatrixProductState.from_fill_fn(
        fill_fn=fill_fn,
        L=L,
        bond_dim=D,
        phys_dim=d,
        cyclic=True,
        shape='lpr',
    )
    return psi
# =====================================================
# Tool Function 4: Multiple Sampling of Local Expectation Values
# =====================================================
def sample_local_expectations_pbc(
    L,
    site,
    phys_dim,
    bond_dim,
    n_samples,
    operator,
    tensor_pool
):
    """
    Sample local expectation values from random complex MPS
    with periodic boundary conditions.
    """
    values = np.empty(n_samples, dtype=np.float64)
    for k in range(n_samples):
        psi =build_pbc_mps_from_pool(L=L,D=bond_dim,d=phys_dim,tensor_pool=tensor_pool)
        # Calculate the normalization coefficient
        norm=psi.H@psi
        # Calculate <psi|X_i|psi>/norm
        psi_X = psi.copy()
        # Apply a local operator on the local site of psi  
        psi_X.gate_(operator, site)
        #For the experiment with the global operator, use the following code to apply the operator to each site of psi
        #for i in range(L):
        #    psi_X.gate_(operator,i)
        
        val = (psi.H@psi_X)
        val_norm=np.real(val)/np.real(norm)
        values[k] =val_norm
    return np.array(values)
# =====================================================
# Experiment A: Changing bond dimension D and physical sites L at fixed d = 2
# =====================================================
def experiment_vary_bond_dimension(L,n_samples):
    site=L//2
    phys_dim = 2
    bond_dims = [2,4,8,16,32,64,128]
    X = qu.pauli('X')
    for D in bond_dims:
        print(f"Generating tensor pool")
        my_pool = generate_haar_tensor_pool(n_pool=2000, D=D, d=phys_dim)
        print(f"Tensor pool has been generated")
        results={}
        start=time.perf_counter()
        data=sample_local_expectations_pbc(
            L=L,
            site=site,
            phys_dim=phys_dim,
            bond_dim=D,
            n_samples=n_samples,
            operator=X,
            tensor_pool=my_pool
        )
        end=time.perf_counter()
        elapsed=end-start
        results[f"D_{D}"] = np.array(data)
        mean = np.mean(data)
        std = np.std(data, ddof=0)
        y = (data - mean) / std
        m4 = np.mean(y**4)
        print(f"k4={m4}")
        print(f"D = {D} finished,elapsed time:{elapsed:.4f}seconds")
        filename = f"../local_D/vary_D_results_local_haar_unitary_L{L}D{D}.npz"
        np.savez(filename, **results)
        print(f"Save to {filename}")
# =====================================================
# Main Program Entry
# =====================================================
if __name__ == "__main__":
    experiment_vary_bond_dimension(L=15,n_samples=200000)