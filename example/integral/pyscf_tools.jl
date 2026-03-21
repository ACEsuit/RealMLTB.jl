module PySCFTools

using LinearAlgebra
using PyCall
using GaussianBasis
using Molecules
using RealMLTB
import GaussianBasis: Acsint

const pyscf = pyimport("pyscf")
const np    = pyimport("numpy")
const dft   = pyimport("pyscf.dft")
const ang2bohr = Acsint.ang2bohr

# =========================
# Utilities
# =========================

function atom_string_angstrom(bs::BasisSet)
    atom_strs = String[]
    for atom in bs.atoms
        push!(atom_strs, "$(atom.Z) $(atom.xyz[1]) $(atom.xyz[2]) $(atom.xyz[3])")
    end
    return join(atom_strs, "; ")
end

function make_pyscf_mol(bs::BasisSet; cart::Bool)
    atom_str = atom_string_angstrom(bs)
    return pyscf.M(atom=atom_str, basis=bs.name, unit="Angstrom", cart=cart)
end

function compute_scaling_factors(S_target::AbstractMatrix, S_source::AbstractMatrix)
    @assert size(S_target) == size(S_source)
    d = sqrt.(diag(S_target) ./ diag(S_source))
    replace!(d, NaN => 0.0)
    replace!(d, Inf => 0.0)
    return d
end

function transform_gb_cart_to_pyscf_sph(M_gb_cart::AbstractMatrix, C::AbstractMatrix, d_c::AbstractVector)
    M_cart_norm = M_gb_cart .* (d_c * d_c')
    return C' * M_cart_norm * C
end

function compute_3c_overlap_grid(mol)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 4
    grids.build()

    ao_value = mol.eval_gto("GTOval", grids.coords)
    weights  = grids.weights
    ao_weighted = ao_value .* weights

    return np.einsum("gi,gj,gk->ijk", ao_weighted, ao_value, ao_value)
end

"""
Return PySCF's cart->sph matrix for the same basis ordering as bs_in.
This is ONLY for testing/diagnostics.
"""
function get_pyscf_cart2sph_coeff(bs_in::BasisSet)
    bs_cart =
        if bs_in.basis[1] isa SphericalShell
            BasisSet(bs_in.name, bs_in.atoms; spherical=false, lib=:acsint)
        else
            bs_in
        end
    mol_sph = make_pyscf_mol(bs_cart; cart=false)
    return mol_sph.cart2sph_coeff()
end

# =========================
# PySCF-backed reference
# =========================

"""
Use Julia Cartesian integrals as source, then apply PySCF's cart->sph matrix.
Useful as a reference chain for debugging the Julia spherical convention.
"""
function compute_spherical_AO_integrals_pyscf(bs_in::BasisSet)
    bs_cart =
        if bs_in.basis[1] isa SphericalShell
            BasisSet(bs_in.name, bs_in.atoms; spherical=false, lib=:acsint)
        else
            bs_in
        end

    mol_cart = make_pyscf_mol(bs_cart; cart=true)
    mol_sph  = make_pyscf_mol(bs_cart; cart=false)

    C = mol_sph.cart2sph_coeff()

    # Cartesian scaling
    S_julia_cart = overlap(bs_cart)
    T_julia_cart = kinetic(bs_cart)

    S_py_cart = mol_cart.intor("int1e_ovlp_cart")
    d_c = compute_scaling_factors(S_py_cart, S_julia_cart)

    # normalized Cartesian
    S_cart_norm = S_julia_cart .* (d_c * d_c')
    T_cart_norm = T_julia_cart .* (d_c * d_c')

    # spherical
    S_sph_raw = C' * S_cart_norm * C
    T_sph_raw = C' * T_cart_norm * C

    d_sph = 1.0 ./ sqrt.(diag(S_sph_raw))
    replace!(d_sph, Inf => 0.0)
    replace!(d_sph, NaN => 0.0)

    S_sph = S_sph_raw .* (d_sph * d_sph')
    T_sph = T_sph_raw .* (d_sph * d_sph')

    # 3-center
    N_cart = bs_cart.nbas
    V_julia_raw = zeros(N_cart, N_cart, N_cart)
    RealMLTB.EPI_1e3c!(V_julia_raw, bs_cart)

    V_cart_norm = zeros(N_cart, N_cart, N_cart)
    @inbounds for k in 1:N_cart, j in 1:N_cart, i in 1:N_cart
        V_cart_norm[i, j, k] = V_julia_raw[i, j, k] * d_c[i] * d_c[j] * d_c[k]
    end

    N_sph = size(C, 2)
    V_tmp1 = reshape(C' * reshape(V_cart_norm, N_cart, :), N_sph, N_cart, N_cart)
    V_p1   = permutedims(V_tmp1, (2, 1, 3))
    V_tmp2 = reshape(C' * reshape(V_p1, N_cart, :), N_sph, N_sph, N_cart)
    V_p2   = permutedims(V_tmp2, (3, 2, 1))
    V_tmp3 = reshape(C' * reshape(V_p2, N_cart, :), N_sph, N_sph, N_sph)
    V_sph_raw = permutedims(V_tmp3, (2, 3, 1))

    V_sph = zeros(N_sph, N_sph, N_sph)
    @inbounds for k in 1:N_sph, j in 1:N_sph, i in 1:N_sph
        V_sph[i, j, k] = V_sph_raw[i, j, k] * d_sph[i] * d_sph[j] * d_sph[k]
    end

    return T_sph, V_sph, S_sph
end

"""
Direct PySCF reference:
- S, T from intor
- V from grid integration
"""
function get_pyscf_reference(bs_in::BasisSet)
    mol_sph = make_pyscf_mol(bs_in; cart=false)
    S_ref = mol_sph.intor("int1e_ovlp_sph")
    T_ref = mol_sph.intor("int1e_kin")
    V_ref = compute_3c_overlap_grid(mol_sph)
    return T_ref, V_ref, S_ref
end

export pyscf,
       make_pyscf_mol,
       get_pyscf_cart2sph_coeff,
       compute_scaling_factors,
       transform_gb_cart_to_pyscf_sph,
       compute_3c_overlap_grid,
       compute_spherical_AO_integrals_pyscf,
       get_pyscf_reference

end # module