#=========================================
#This piece of code implements: numerical calculation of the Weingarten factors, and then derives the theoretical values according to our theory.
#=========================================
import itertools
import sympy as sp
import numpy as np
# ------------------------------------------------------------
# 1. Automatically Generate Symmetric Groups
# ------------------------------------------------------------
def permutations(k):
    return list(itertools.permutations(range(k)))

# ------------------------------------------------------------
# 2. Cycle Type and Number of Cycles
# ------------------------------------------------------------
def cycle_type(p):
    k = len(p)
    visited = [False] * k
    cycles = []
    for i in range(k):
        if not visited[i]:
            c = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = p[j]
                c += 1
            cycles.append(c)
    return tuple(sorted(cycles, reverse=True))

def cycle_count(p):
    return len(cycle_type(p))

# ------------------------------------------------------------
# 3. Permutation Operations: Inverse and Composition
# ------------------------------------------------------------
def perm_inverse(p):
    k = len(p)
    inv = [0]*k
    for i in range(k):
        inv[p[i]] = i
    return tuple(inv)

def perm_comp(p, q):
    return tuple(p[q[i]] for i in range(len(p)))
#3.5Sort according to conjugacy classes
def sort_perms_by_conjugacy_class(k):
    perms = permutations(k)

    def key_func(p):
        ctype = cycle_type(p)      # 比如 (2,1,1)
        cycles = len(ctype)        # 循环数：例如 #(2,1,1) = 3
        return (-cycles, ctype, p) # 循环数取负号实现“从大到小”

    return sorted(perms, key=key_func)
# ------------------------------------------------------------
# 4. Construct a Gram matrix (supports arbitrary symbolic expressions d_expr)
# ------------------------------------------------------------
def gram_matrix_symbolic(k, d_expr):
    perms = sort_perms_by_conjugacy_class(k)
    n = len(perms)
    G = sp.zeros(n, n)

    for i, sigma in enumerate(perms):
        inv_sigma = perm_inverse(sigma)
        for j, tau in enumerate(perms):
            comp = perm_comp(inv_sigma, tau)
            G[i, j] = d_expr ** cycle_count(comp)

    return G, perms
# ------------------------------------------------------------
# 4.5 Construct an np-type Gram matrix
# ------------------------------------------------------------    
def gram_matrix_np(k,d_np,D_np):
    d, D = sp.symbols("d D", real=True)
    d_expr = d * D
    G,perms=gram_matrix_symbolic(k,d_expr)
    G=G.subs({d:d_np,D:D_np})
    G_np = np.array(G.tolist()).astype(np.float64)
    return G_np
# ------------------------------------------------------------
# 5. Construct the Weingarten matrix (sp type)
# ------------------------------------------------------------
def weingarten_matrix_symbolic(k, d_expr):
    G, perms = gram_matrix_symbolic(k, d_expr)
    W = G.inv()
    return W, G, perms
# ------------------------------------------------------------
# 5.5 Construct the Weingarten matrix (np type)
# ------------------------------------------------------------
def weingarten_matrix_np(k,d_np,D_np):
    G_np=gram_matrix_np(k,d_np,D_np)
    W_np=np.linalg.inv(G_np)
    return W_np
# ------------------------------------------------------------
# 6 Construct an observable matrix (sp type)
# ------------------------------------------------------------
def diag_matrix_even_cycles_only(k, d_expr):
    """
Return a diagonal matrix M (k! × k!) whose indexing order is consistent with sort_perms_by_conjugacy_class(k). If the cycle decomposition of σ contains no odd-length cycles (that is, each length in cycle_type(σ) is even), then M[i,i] = d_expr ** #(σ); otherwise, it is 0. The identity permutation e is automatically excluded (because it contains a 1-cycle), and it is explicitly excluded here as well.
    """
    perms = sort_perms_by_conjugacy_class(k)
    n = len(perms)
    M = sp.zeros(n, n)

    e = tuple(range(k))

    def has_no_odd_length_cycles(sigma):
        return all(L % 2 == 0 for L in cycle_type(sigma))

    for i, sigma in enumerate(perms):
        if sigma != e and has_no_odd_length_cycles(sigma):
            M[i, i] = d_expr ** cycle_count(sigma)
        # else: keep 0
    return M, perms
# ------------------------------------------------------------
# 6.5 Constructing an observable matrix (np type)
# ------------------------------------------------------------
def diag_matrix_even_cycles_only_np(k, d_np):
    d = sp.symbols("d", real=True)
    O,perms=diag_matrix_even_cycles_only(k=k,d_expr=d)
    O=O.subs({d:d_np})
    O_np = np.array(O.tolist()).astype(np.float64)
    return O_np
# ------------------------------------------------------------
# 7 Construct a diagonal matrix at unobserved sites (sp type)
# ------------------------------------------------------------
def diag_matrix(k, d_expr):
    """
    Return the diagonal matrix M (k! × k!), whose indexing order is consistent with sort_perms_by_conjugacy_class(k). The diagonal elements are d^#(σ).
    """
    perms = sort_perms_by_conjugacy_class(k)
    n = len(perms)
    M = sp.zeros(n, n)
    for i, sigma in enumerate(perms):
            M[i, i] = d_expr ** cycle_count(sigma)
        # else: keep 0
    return M, perms
# ------------------------------------------------------------
# 7.5 Construct a diagonal matrix at unobserved sites (np type)
# ------------------------------------------------------------
def diag_matrix_np(k,d_np):
    d = sp.symbols("d", real=True)
    Diag,perms=diag_matrix(k=k, d_expr=d)
    Diag=Diag.subs({d:d_np})
    Diag_np = np.array(Diag.tolist()).astype(np.float64)  
    return Diag_np
# ------------------------------------------------------------
# Calculating the k-th Moment of <psi|_mps \hat{O} |psi>_mps
# ------------------------------------------------------------
#Experiment 1, the observable in MPS is local, d=2, change k and D, L (number of sites), and calculate the k-th moment
#The output file is filename_k4=f"haar_moment2/haar_local_L_D_k4" and filename_k2=f"haar_moment2/haar_local_L_D_k2"
if __name__ == "__main__":
    # Set the order of the symmetric group k, that is, the order of the k-th moment
    k=[2,4]
    L=[12,13,14,15,16]
    D_np=[2,4,8,16,32,64,128]
    dd=2
    Mk = np.zeros((len(k), len(L), len(D_np)), dtype=np.float64) 
    results_k4={}
    results_k2={}
    for i_idx,i in enumerate(k):
        for l_idx,l in enumerate(L):
            for j_idx,j in enumerate(D_np):
                #Compute the Weingarten matrix of np type under the corresponding parameters
                W_np=weingarten_matrix_np(k=i,d_np=dd,D_np=j)
                #Compute the operator matrix of type np under the corresponding parameters
                O_np=diag_matrix_even_cycles_only_np(k=i, d_np=dd)
                #np type Gram matrix
                G_np=gram_matrix_np(k=i,d_np=1,D_np=j)
                #np type Operator matrix in non-measurement sites
                Diag_np=diag_matrix_np(k=i,d_np=dd)
                #Complete matrix corresponding to the measurement site
                TO=W_np@O_np@G_np
                #Complete matrix corresponding to non-measured sites
                T=W_np@Diag_np@G_np
                #Multiply these matrices and take the trace
                mps=TO
                mps_norm=T
                for _ in range(l-1):mps=mps@T
                for _ in range(l-1):mps_norm=mps_norm@T
                Mk[i_idx,l_idx,j_idx]=mps.trace()/mps_norm.trace()
    for l_idx,l in enumerate(L):
        results_k4[f"L_{l}"]=Mk[1,l_idx,:]/(Mk[0,l_idx,:]**2)
    filename_k4=f"haar_moment2/haar_local_L_D_k4"
    np.savez(filename_k4,**results_k4)
    for l_idx,l in enumerate(L):
        results_k2[f"L_{l}"]=Mk[0,l_idx,:]
    filename_k2=f"haar_moment2/haar_local_L_D_k2"
    np.savez(filename_k2,**results_k2)    
    for i in range(len(L)):
        for j in range(len(D_np)):
            k4=Mk[1,i,j]/((Mk[0,i,j])**2)
            print(f'when L={L[i]},d={dd},D={D_np[j]}，k4={k4}')
