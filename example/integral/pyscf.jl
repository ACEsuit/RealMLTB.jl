using LinearAlgebra
using PyCall
using GaussianBasis
using Molecules

# --- 1. Normalization Factor Calculation ---
"""
    compute_scaling_factors(S_target, S_source)

Compute the scaling factors `d` such that `S_target ≈ D * S_source * D`.
Returns: Vector{Float64} (Diagonal elements)
"""
function compute_scaling_factors(S_target::AbstractMatrix, S_source::AbstractMatrix)
    # Ensure dimensions match
    @assert size(S_target) == size(S_source)
    # Calculate square root of ratio: d_i = sqrt(S_target_ii / S_source_ii)
    d = sqrt.(diag(S_target) ./ diag(S_source))
    # Handle potential NaNs (though diagonal shouldn't be zero)
    replace!(d, NaN => 0.0)
    replace!(d, Inf => 0.0)
    return d
end

# --- 2. General Matrix Transformation ---
"""
    transform_matrix(M, L, R)

General matrix transformation: out = L' * M * R.
Used for Basis transformation (C' * S * C) or Normalization (D * S * D).
"""
function transform_matrix(M::AbstractMatrix, L::AbstractMatrix, R::AbstractMatrix)
    # Optimized multiplication order: (L' * M) * R
    return (L' * M) * R
end

# Overload: Symmetric transformation (M -> A' M A)
transform_matrix(M::AbstractMatrix, A::AbstractMatrix) = transform_matrix(M, A, A)

# --- 3. Core Logic: Raw Cartesian -> Raw Spherical ---
"""
    convert_gb_cart_to_sph(M_cart_raw, C_pyscf, d_c, d_s)

Convert raw Cartesian integrals (from GaussianBasis.jl) to raw Spherical integrals.
Formula: S_sph_raw = D_s^-1 * C' * (D_c * S_cart_raw * D_c) * C * D_s^-1
"""
function convert_gb_cart_to_sph(
    M_cart_raw::AbstractMatrix, # Input: GB Raw Cartesian Matrix
    C_pyscf::AbstractMatrix,    # Input: PySCF Transformation Matrix
    d_c::AbstractVector,        # Input: Cartesian Scaling Factors (GB->PySCF)
    d_s::AbstractVector         # Input: Spherical Scaling Factors (GB->PySCF)
)
    # 1. Cartesian Normalization: M_norm = D_c * M_raw * D_c
    #    Using broadcasting is faster than matrix multiplication for diagonal scaling
    #    M_norm[i,j] = M_raw[i,j] * d_c[i] * d_c[j]
    M_cart_norm = M_cart_raw .* (d_c * d_c')
    
    # 2. Cartesian -> Spherical Transform: M_sph_norm = C' * M_cart_norm * C
    M_sph_norm = transform_matrix(M_cart_norm, C_pyscf)
    
    # 3. Denormalization (Return to GB Raw Spherical standard): M_sph_raw = D_s^-1 * M_sph_norm * D_s^-1
    #    Equivalent to dividing by the scaling factors
    M_sph_raw = M_sph_norm ./ (d_s * d_s')
    
    return M_sph_raw
end

# --- 4. 3-Center Integral Tensor Transformation ---
"""
    transform_3c_tensor_full(V_cart, C, d_c, d_s)

Transform ALL three indices (i, j, k) of a 3-center tensor from Cartesian to Spherical.
Steps: Normalization -> Tensor Contraction (i,j,k) -> Denormalization
"""
function transform_3c_tensor_full(
    V_cart::AbstractArray{T, 3}, 
    C::AbstractMatrix, 
    d_c::AbstractVector, 
    d_s::AbstractVector
) where T
    ni_c, nj_c, nk_c = size(V_cart)
    ni_s, nj_s, nk_s = size(C, 2), size(C, 2), size(C, 2)
    
    V_temp1 = zeros(T, ni_c, nj_c, nk_c)
    @inbounds for k in 1:nk_c, j in 1:nj_c, i in 1:ni_c
        V_temp1[i,j,k] = V_cart[i,j,k] * d_c[i] * d_c[j] * d_c[k]
    end

    # V_temp2[a, j, k] = sum_i C[i, a] * V_temp1[i, j, k]
    V_flat1 = reshape(V_temp1, ni_c, nj_c * nk_c)
    V_step1_flat = C' * V_flat1
    V_step1 = reshape(V_step1_flat, ni_s, nj_c, nk_c)

    V_perm1 = permutedims(V_step1, (2, 1, 3)) # (j, i_s, k)
    V_flat2 = reshape(V_perm1, nj_c, ni_s * nk_c)
    V_step2_flat = C' * V_flat2
    V_step2 = reshape(V_step2_flat, nj_s, ni_s, nk_c) # (j_s, i_s, k_c)

    V_perm2 = permutedims(V_step2, (3, 2, 1)) # (k_c, i_s, j_s)
    V_flat3 = reshape(V_perm2, nk_c, ni_s * nj_s)
    V_step3_flat = C' * V_flat3
    V_step3 = reshape(V_step3_flat, nk_s, ni_s, nj_s) # (k_s, i_s, j_s)
    V_final_norm = permutedims(V_step3, (2, 3, 1))
    V_out = zeros(T, ni_s, nj_s, nk_s)
    @inbounds for k in 1:nk_s, j in 1:nj_s, i in 1:ni_s
        V_out[i,j,k] = V_final_norm[i,j,k] / (d_s[i] * d_s[j] * d_s[k])
    end
    
    return V_out
end

"""
    transform_gb_cart_to_pyscf_sph(M_gb_cart, C, d_c)

Transform unnormalized Cartesian integrals (from GaussianBasis.jl) directly to 
normalized Spherical integrals (matching PySCF's output).

Formula: M_py_sph = C' * (D_c * M_gb_cart * D_c) * C

Args:
    M_gb_cart: Raw Cartesian matrix from GaussianBasis.jl
    C:         Transformation matrix (Cart -> Sph) from PySCF
    d_c:       Cartesian scaling factors (GB -> PySCF)
"""
function transform_gb_cart_to_pyscf_sph(M_gb_cart::AbstractMatrix, C::AbstractMatrix, d_c::AbstractVector)
    # 1. Normalize Cartesian: M_norm = D_c * M_raw * D_c
    #    Using broadcasting for efficiency: M_ij * d_i * d_j
    M_cart_norm = M_gb_cart .* (d_c * d_c')
    
    # 2. Transform to Spherical: M_sph = C' * M_cart_norm * C
    #    Note: (C' * M) * C is generally faster
    M_pyscf_sph = C' * M_cart_norm * C
    
    return M_pyscf_sph
end

# --- Core Technique: Use DFT Grid for Numerical Reference Integration ---
function compute_3c_overlap_grid(mol)
    # 1. Generate high-precision integration grid (Level 3 or 4 is usually sufficient)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 4 
    grids.build()
    
    # 2. Evaluate Atomic Orbitals (AO) on grid points. Shape: (N_grid, N_basis)
    #    Note: PySCF returns normalized AOs by default!
    ao_value = mol.eval_gto("GTOval", grids.coords)
    weights = grids.weights
    
    # 3. Perform numerical integration: sum_g w_g * phi_i * phi_j * phi_k
    #    Use numpy.einsum for efficient tensor contraction.
    #    Formula: V[i,j,k] = sum(g) weights[g] * ao[g,i] * ao[g,j] * ao[g,k]
    
    # To optimize memory usage, pre-multiply weights into one of the AO arrays
    ao_weighted = ao_value .* weights
    
    # Compute tensor contraction: (N_grid, N_bas) -> (N_bas, N_bas, N_bas)
    V_grid = np.einsum("gi,gj,gk->ijk", ao_weighted, ao_value, ao_value)
    
    return V_grid
end

