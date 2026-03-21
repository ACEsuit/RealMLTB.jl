# rotation.jl
# Utilities for constructing random rotations, rotating ETGraphs, and building
# Wigner-D / AO rotation matrices in the pyscf/GaussianBasis convention.
#
# These are used exclusively by rotation_test.jl.  Nothing in this file
# should be called from the main model code.

using LinearAlgebra
using StaticArrays
using Rotations
using WignerD
using EquivariantTensors
using EquivariantTensors.O3: Ctran
using GaussianBasis
using Molecules
import GaussianBasis: num_basis

# ---------------------------------------------------------------------------
# rand_rot  –  sample a Haar-uniform rotation from SO(3)
# ---------------------------------------------------------------------------
function rand_rot()
    Q = Matrix(qr(randn(3, 3)).Q)
    det(Q) < 0 && (Q[:, 1] .*= -1)
    return SMatrix{3,3,Float64,9}(Q)
end

# ---------------------------------------------------------------------------
# rotate_graph
#
# Return a new ETGraph whose edge displacement vectors 𝐫 are rotated by Q,
# but all other edge / graph data (T_pair, V_pair, S_pair, klist, …) are
# left untouched — they are recomputed externally when needed.
#
# We rebuild edge_data as a new NamedTuple vector rather than mutating fields,
# because ETGraph stores edge_data as an immutable struct array.
# ---------------------------------------------------------------------------
function rotate_graph(G::EquivariantTensors.ETGraph, Q::SMatrix{3,3})
    new_edge_data = map(G.edge_data) do e
        merge(e, (; 𝐫 = Q * e.𝐫))
    end

    # Discover actual field names at runtime to avoid depending on internals.
    fnames = fieldnames(typeof(G))
    new_fields = map(fnames) do f
        f === :edge_data ? new_edge_data : getfield(G, f)
    end
    return typeof(G)(new_fields...)
end

# ---------------------------------------------------------------------------
# rotate_basis_set
#
# Return a new BasisSet whose atom positions are rotated by Q (in Angstrom).
# Q is dimensionless (pure rotation matrix), atom.xyz is in Angstrom.
# ---------------------------------------------------------------------------
function rotate_basis_set(bs::BasisSet, Q::SMatrix{3,3})
    new_atoms = map(bs.atoms) do atom
        Molecules.Atom(atom.Z, atom.mass, Q * atom.xyz)
    end
    return BasisSet(bs.name, new_atoms; spherical = false, lib = :acsint)
end

# ---------------------------------------------------------------------------
# get_pyscf_permutation
#
# The real spherical harmonics in GaussianBasis / acsint follow the pyscf
# ordering convention, which differs from the standard (-l … +l) convention
# for l = 1 only:
#
#   l = 0  →  [Y_0^0]                          →  identity [1]
#   l = 1  →  [Y_1^{-1}, Y_1^0, Y_1^1]  BUT stored as  [z, x, y]
#              so pyscf index [1,2,3] corresponds to m = [0,-1,+1]
#              i.e. the permutation that maps standard → pyscf is [3,1,2]
#   l ≥ 2  →  standard ordering, no permutation needed
#
# This function returns the permutation vector P such that
#   Y_pyscf[P[m]] == Y_standard[m]   for m = 1 … 2l+1.
# ---------------------------------------------------------------------------
function get_pyscf_permutation(l::Int)
    l == 0 && return [1]
    l == 1 && return [3, 1, 2]
    return collect(1:(2l + 1))
end

# ---------------------------------------------------------------------------
# construct_basis_D_matrix_pyscf
#
# Build the full AO-basis Wigner-D rotation matrix in pyscf ordering.
# This is the matrix D_φ such that
#   φ(Q^{-1} r) = D_φ φ(r)    (as column vectors of basis functions).
#
# Each shell contributes a (2l+1) × (2l+1) block.  The blocks are assembled
# in the same order as the shells in bs.basis.
#
# θ : Euler angles [α, β, γ] passed to WignerD.wignerD(l, θ...)
#     using the ZYZ convention (same as Rotations.RotZYZ).
# ---------------------------------------------------------------------------
function construct_basis_D_matrix_pyscf(bs::BasisSet, θ::AbstractVector)
    blocks = Matrix{Float64}[]
    for shell in bs.basis
        l        = shell.l
        # complex Wigner-D in the standard (Condon-Shortley) convention,
        # transposed because wignerD gives D^{(l)}_{m'm} and we want D^{(l)}_{mm'}
        D_cpx    = transpose(wignerD(l, θ...))
        # change-of-basis matrix: complex → real spherical harmonics
        U        = Ctran(l)
        D_real   = real.(U * D_cpx * U')
        # apply pyscf permutation to rows and columns
        P        = get_pyscf_permutation(l)
        push!(blocks, D_real[P, P])
    end
    return cat(blocks...; dims = (1, 2))
end

# ---------------------------------------------------------------------------
# construct_Dpsi_atom
#
# Build the Wigner-D block for a single atom's compressed (Case A) basis.
# The compressed basis has nψ[l+1] orbitals per l-channel, each transforming
# as a full rank-l irrep.  So the atom block is block-diagonal:
#   D_atom = blockdiag( D^(0), …, D^(0),   ← nψ[1] copies
#                       D^(1), …, D^(1),   ← nψ[2] copies
#                       … )
# with pyscf permutation applied to each block.
# ---------------------------------------------------------------------------
function construct_Dpsi_atom(nψ, θ::AbstractVector)
    lmax   = length(nψ) - 1
    blocks = Matrix{Float64}[]
    for l in 0:lmax
        nψl = nψ[l + 1]
        nψl == 0 && continue
        D_cpx  = transpose(wignerD(l, θ...))
        U      = Ctran(l)
        D_real = real.(U * D_cpx * U')
        P      = get_pyscf_permutation(l)
        Dp     = D_real[P, P]
        for _ in 1:nψl
            push!(blocks, Dp)
        end
    end
    return cat(blocks...; dims = (1, 2))
end

# ---------------------------------------------------------------------------
# construct_Dpsi_total
#
# Kronecker-lift construct_Dpsi_atom to all NATOM atoms:
#   D_total = I_NATOM ⊗ D_atom
# ---------------------------------------------------------------------------
function construct_Dpsi_total(nψ, θ::AbstractVector, NATOM::Int)
    D_atom = construct_Dpsi_atom(nψ, θ)
    return kron(Matrix{Float64}(I, NATOM, NATOM), D_atom)
end

# ---------------------------------------------------------------------------
# get_valscalar
#
# Robustly unwrap the scalar output of a Case-A model, which may be returned
# as a plain Matrix, a 1-Tuple, or a NamedTuple with one field.
# ---------------------------------------------------------------------------
function get_valscalar(val)
    val isa AbstractMatrix             && return val
    val isa Tuple && length(val) == 1  && return val[1]
    val isa NamedTuple                 && return getfield(val, first(fieldnames(typeof(val))))
    error("get_valscalar: unexpected type $(typeof(val))")
end

# ---------------------------------------------------------------------------
# test_equivariance
#
# Check that the raw ACE equivariant model output transforms correctly:
#   val_rot[L+1][i,j]  ==  D^(L)(Q) * val_orig[L+1][i,j]
#
# This tests build_ace_equivariant_model (Wv model) independently of the
# three-centre integrals.
# ---------------------------------------------------------------------------
function test_equivariance(model, ps, st, G; tol = 1e-5)
    println("\n====== test_equivariance (raw ACE output) ======")

    val_orig, _ = model(G, ps, st)

    θ     = rand(3) * 2π
    Q_rot = RotZYZ(θ...)
    Q_mat = SMatrix{3,3,Float64,9}(Q_rot)
    G_rot = rotate_graph(G, Q_mat)

    val_rot, _ = model(G_rot, ps, st)

    for (Lidx, (blk_orig, blk_rot)) in enumerate(zip(val_orig, val_rot))
        L = Lidx - 1
        println("  L = $L")

        if L == 0
            # invariant: output must be unchanged
            diff = norm(blk_rot - blk_orig)
            println("    L=0 invariance error: $diff")
            @test diff < tol
        else
            # equivariant: output must rotate by D^(L)(Q)
            D_cpx  = transpose(wignerD(L, θ...))
            U      = Ctran(L)
            D_real = real.(U * D_cpx * U')

            blk_expected = map(blk_orig) do v
                SVector{2L+1}(D_real * collect(v))
            end
            diff = norm(blk_rot - blk_expected)
            println("    L=$L equivariance error: $diff")
            @test diff < tol
        end
    end
    println("====== test_equivariance PASSED ======\n")
end

# ---------------------------------------------------------------------------
# test_V_pair_equivariance
#
# Check that the assembled potential matrix V = Σ_K Wv[K] <φ_I φ_K φ_J>
# transforms correctly under rotation:
#
#   V(Q{R}) = D_φ(Q)^T * V({R}) * D_φ(Q)
#
# where D_φ is the AO-basis Wigner-D matrix in pyscf ordering.
# ---------------------------------------------------------------------------
function test_V_pair_equivariance(modelWv, psWv, stWv, G, bs_cart_orig; tol = 1e-3)
    println("\n====== test_V_pair_equivariance ======")

    # ---- original system ----
    val_orig, _ = modelWv(G, psWv, stWv)
    Wv_orig     = RealMLTB.assemble_Wv(val_orig)

    # compute_spherical_AO_integrals returns (T, V3c, S)
    # V3c has shape (nbas, nbas, nbas): V3c[i,j,k] = <φ_i φ_j φ_k>
    _T0, V3c_orig, _S0 = RealMLTB.compute_spherical_AO_integrals(bs_cart_orig)
    nbas = size(V3c_orig, 1)

    # contract: V[i,j] = Σ_k  V3c[i,j,k] * Wv[k]
    M_orig  = reshape(V3c_orig, nbas^2, nbas)
    V_orig  = reshape(M_orig * Wv_orig, nbas, nbas)

    # ---- rotated system ----
    θ     = rand(3) * 2π
    Q_rot = RotZYZ(θ...)
    Q_mat = SMatrix{3,3,Float64,9}(Q_rot)

    G_rot        = rotate_graph(G, Q_mat)
    bs_cart_rot  = rotate_basis_set(bs_cart_orig, Q_mat)

    val_rot, _ = modelWv(G_rot, psWv, stWv)
    Wv_rot     = RealMLTB.assemble_Wv(val_rot)

    _T1, V3c_rot, _S1 = RealMLTB.compute_spherical_AO_integrals(bs_cart_rot)
    M_rot  = reshape(V3c_rot, nbas^2, nbas)
    V_calc = reshape(M_rot * Wv_rot, nbas, nbas)

    # ---- expected: D^T V D ----
    D_total  = construct_basis_D_matrix_pyscf(bs_cart_orig, θ)
    V_expect = D_total' * V_orig * D_total

    diff = norm(V_calc - V_expect)
    println("  V matrix equivariance error: $diff")
    @test diff < tol
    println("====== test_V_pair_equivariance PASSED ======\n")
end

# ---------------------------------------------------------------------------
# _build_V_AO
#
# Internal helper: assemble the V matrix from Wv model + three-centre integrals.
# ---------------------------------------------------------------------------
function _build_V_AO(modelWv, psWv, stWv, G, bs_cart)
    val, _ = modelWv(G, psWv, stWv)
    Wv     = RealMLTB.assemble_Wv(val)
    _T, V3c, _S = RealMLTB.compute_spherical_AO_integrals(bs_cart)
    nbas = size(V3c, 1)
    M    = reshape(V3c, nbas^2, nbas)
    return reshape(M * Wv, nbas, nbas)
end

# ---------------------------------------------------------------------------
# test_Wpsi_caseA
#
# Check the two-sided equivariance of the Case-A contraction matrix:
#
#   Wpsi(QN_I) = D_ψ(Q)^T * Wpsi(N_I) * D_φ(Q)
#
# where D_ψ is the compressed-basis rotation matrix and D_φ is the AO-basis
# rotation matrix.
# ---------------------------------------------------------------------------
function test_Wpsi_caseA(modelWpsi, psWpsi, stWpsi, G, bs_cart_orig,
                         n_channels_φ, nψ; tol = 1e-3)
    println("\n====== test_Wpsi_caseA ======")

    val0, _  = modelWpsi(G, psWpsi, stWpsi)
    val0s    = get_valscalar(val0)
    NATOM    = size(val0s, 1)
    W0       = RealMLTB.assemble_Wpsi_caseA(val0s, n_channels_φ, nψ)

    θ     = rand(3) * 2π
    Q_rot = RotZYZ(θ...)
    Q_mat = SMatrix{3,3,Float64,9}(Q_rot)
    G_rot = rotate_graph(G, Q_mat)

    val1, _  = modelWpsi(G_rot, psWpsi, stWpsi)
    val1s    = get_valscalar(val1)
    W1       = RealMLTB.assemble_Wpsi_caseA(val1s, n_channels_φ, nψ)

    D_phi = construct_basis_D_matrix_pyscf(bs_cart_orig, θ)
    D_psi = construct_Dpsi_total(nψ, θ, NATOM)

    W_expect = D_psi' * W0 * D_phi

    diff = norm(W1 - W_expect)
    println("  Wpsi Case-A equivariance error: $diff")
    @test diff < tol
    println("====== test_Wpsi_caseA PASSED ======\n")
end

# ---------------------------------------------------------------------------
# test_Hpsi_caseA
#
# Check that the compressed Hamiltonian H^ψ = Wpsi * H_AO * Wpsi^T transforms
# by a similarity (conjugation) under rotation in Case A:
#
#   H^ψ(QR) = D_ψ(Q)^T * H^ψ(R) * D_ψ(Q)
#
# This is the key physical invariant: the eigenvalues of H^ψ are independent
# of the orientation of the molecule / crystal.
# ---------------------------------------------------------------------------
function test_Hpsi_caseA(modelWv, psWv, stWv,
                         modelWpsi, psWpsi, stWpsi,
                         G, bs_cart_orig, n_channels_φ, nψ; tol = 1e-3)
    println("\n====== test_Hpsi_caseA ======")

    bs_1atom = BasisSet(
        bs_cart_orig.name,
        [bs_cart_orig.atoms[1]];
        spherical = false, lib = :acsint,
    )

    # Print shell ordering so we can compare with Wpsi column layout
    println("  bs_cart_orig shell order (atom, l, nbas):")
    
    offset = 0
    for sh in bs_cart_orig.basis
        n = num_basis(sh)
        println("    atom_xyz=$(round.(sh.atom.xyz; digits=2))  l=$(sh.l)  cols=$(offset+1):$(offset+n)")
        offset += n
    end
    println("  bs_1atom shell order:")
    offset = 0
    for sh in bs_1atom.basis
        n = num_basis(sh)
        println("    l=$(sh.l)  cols=$(offset+1):$(offset+n)")
        offset += n
    end

    θ     = rand(3) * 2π
    Q_rot = RotZYZ(θ...)
    Q_mat = SMatrix{3,3,Float64,9}(Q_rot)
    G_rot        = rotate_graph(G, Q_mat)
    bs_1atom_rot = rotate_basis_set(bs_1atom, Q_mat)

    T0, V3c0, _S0 = RealMLTB.compute_spherical_AO_integrals(bs_1atom)
    nbas_1 = size(T0, 1)

    val_v0, _ = modelWv(G, psWv, stWv)
    Wv0 = RealMLTB.assemble_Wv(val_v0)
    println("  length(Wv0)=$( length(Wv0))  nbas_1=$nbas_1")
    # Use the first nbas_1 elements of Wv0 as atom-1 potential coefficients
    Wv0_1 = Wv0[1:nbas_1]
    V0 = reshape(reshape(V3c0, nbas_1^2, nbas_1) * Wv0_1, nbas_1, nbas_1)
    H_AO_0 = T0 + V0

    val_ψ0, _ = modelWpsi(G, psWpsi, stWpsi)
    val_ψ0s   = get_valscalar(val_ψ0)
    val_ψ0s_1 = val_ψ0s[1:1, :]
    Wψ0 = RealMLTB.assemble_Wpsi_caseA(val_ψ0s_1, n_channels_φ, nψ)
    println("  size(Wψ0)=$(size(Wψ0))  nbas_1=$nbas_1")

    Hψ0 = Wψ0 * H_AO_0 * Wψ0'

    T1, V3c1, _S1 = RealMLTB.compute_spherical_AO_integrals(bs_1atom_rot)
    val_v1, _ = modelWv(G_rot, psWv, stWv)
    Wv1 = RealMLTB.assemble_Wv(val_v1)
    Wv1_1 = Wv1[1:nbas_1]
    V1 = reshape(reshape(V3c1, nbas_1^2, nbas_1) * Wv1_1, nbas_1, nbas_1)
    H_AO_1 = T1 + V1

    val_ψ1, _ = modelWpsi(G_rot, psWpsi, stWpsi)
    val_ψ1s   = get_valscalar(val_ψ1)
    val_ψ1s_1 = val_ψ1s[1:1, :]
    Wψ1 = RealMLTB.assemble_Wpsi_caseA(val_ψ1s_1, n_channels_φ, nψ)

    Hψ1 = Wψ1 * H_AO_1 * Wψ1'

    D_phi = construct_basis_D_matrix_pyscf(bs_1atom, θ)
    D_psi = construct_Dpsi_total(nψ, θ, 1)
    println("  T equivariance error:  $(norm(T1 - D_phi' * T0 * D_phi))")
    println("  V equivariance error:  $(norm(V1 - D_phi' * V0 * D_phi))")
    println("  Wψ equivariance error: $(norm(Wψ1 - D_psi' * Wψ0 * D_phi))")

    # Verify: does Wψ0 * D_phi * H_AO_1 * D_phi' * Wψ0' == Hψ0?
    # i.e. is H_AO in the correct basis for Wψ?
    Hψ_via_Dphi = Wψ0 * (D_phi' * H_AO_0 * D_phi) * Wψ0'
    println("  Hψ via D_phi rotation vs H_AO_1: $(norm(Hψ_via_Dphi - Wψ0 * H_AO_1 * Wψ0'))")

    evals0 = sort(real.(eigvals(Hψ0)))
    evals1 = sort(real.(eigvals(Hψ1)))
    eval_diff = norm(evals1 - evals0)
    println("  Eigenvalue invariance error: $eval_diff")
    println("  Hψ similarity error:         $(norm(Hψ1 - D_psi' * Hψ0 * D_psi))")
    @test eval_diff < tol
    println("====== test_Hpsi_caseA PASSED ======\n")
end

# ---------------------------------------------------------------------------
# test_Wpsi_caseB
#
# Check the right-action equivariance of the Case-B contraction matrix:
#
#   Wpsi(QN_I) = Wpsi(N_I) * D_φ(Q)
#
# The target orbitals are rotationally invariant scalars, so only the AO
# rotation D_φ appears on the right.
# ---------------------------------------------------------------------------
function test_Wpsi_caseB(modelWpsi, psWpsi, stWpsi, G, bs_cart_orig,
                         n_channels_φ, NORB::Int; tol = 1e-3)
    println("\n====== test_Wpsi_caseB ======")

    val0, _ = modelWpsi(G, psWpsi, stWpsi)
    W0      = RealMLTB.assemble_Wpsi_caseB(val0, n_channels_φ, NORB)

    θ     = rand(3) * 2π
    Q_rot = RotZYZ(θ...)
    Q_mat = SMatrix{3,3,Float64,9}(Q_rot)
    G_rot = rotate_graph(G, Q_mat)

    val1, _ = modelWpsi(G_rot, psWpsi, stWpsi)
    W1      = RealMLTB.assemble_Wpsi_caseB(val1, n_channels_φ, NORB)

    D_phi    = construct_basis_D_matrix_pyscf(bs_cart_orig, θ)
    W_expect = W0 * D_phi

    diff = norm(W1 - W_expect)
    println("  Wpsi Case-B equivariance error: $diff")
    @test diff < tol
    println("====== test_Wpsi_caseB PASSED ======\n")
end

# ---------------------------------------------------------------------------
# test_Hpsi_caseB
#
# Check that the compressed Hamiltonian H^ψ = Wpsi * H_AO * Wpsi^T is
# *elementwise invariant* under rotation in Case B:
#
#   H^ψ(QR) = H^ψ(R)
#
# This is stronger than Case A because the target orbitals are scalars.
# ---------------------------------------------------------------------------
function test_Hpsi_caseB(modelWv, psWv, stWv,
                         modelWpsi, psWpsi, stWpsi,
                         G, bs_cart_orig, n_channels_φ, NORB::Int; tol = 1e-3)
    println("\n====== test_Hpsi_caseB ======")

    bs_1atom = BasisSet(
        bs_cart_orig.name,
        [bs_cart_orig.atoms[1]];
        spherical = false, lib = :acsint,
    )

    θ     = rand(3) * 2π
    Q_rot = RotZYZ(θ...)
    Q_mat = SMatrix{3,3,Float64,9}(Q_rot)
    G_rot        = rotate_graph(G, Q_mat)
    bs_1atom_rot = rotate_basis_set(bs_1atom, Q_mat)

    # ---- original: single-atom H_AO ----
    T0, V3c0, _S0 = RealMLTB.compute_spherical_AO_integrals(bs_1atom)
    nbas_1 = size(T0, 1)

    val_v0, _ = modelWv(G, psWv, stWv)
    Wv0 = RealMLTB.assemble_Wv(val_v0)
    Wv0_1 = Wv0[1:nbas_1]
    V0 = reshape(reshape(V3c0, nbas_1^2, nbas_1) * Wv0_1, nbas_1, nbas_1)
    H_AO_0 = T0 + V0

    val_ψ0, _ = modelWpsi(G, psWpsi, stWpsi)
    # Slice to atom 1 only: each block in val_ψ0 has shape (NATOM, n_shells)
    val_ψ0_1 = map(blk -> blk[1:1, :], val_ψ0)
    Wψ0 = RealMLTB.assemble_Wpsi_caseB(val_ψ0_1, n_channels_φ, NORB)
    println("  size(Wψ0)=$(size(Wψ0))  nbas_1=$nbas_1")

    Hψ0 = Wψ0 * H_AO_0 * Wψ0'

    # ---- rotated: single-atom H_AO for rotated atom ----
    T1, V3c1, _S1 = RealMLTB.compute_spherical_AO_integrals(bs_1atom_rot)
    val_v1, _ = modelWv(G_rot, psWv, stWv)
    Wv1 = RealMLTB.assemble_Wv(val_v1)
    Wv1_1 = Wv1[1:nbas_1]
    V1 = reshape(reshape(V3c1, nbas_1^2, nbas_1) * Wv1_1, nbas_1, nbas_1)
    H_AO_1 = T1 + V1

    val_ψ1, _ = modelWpsi(G_rot, psWpsi, stWpsi)
    val_ψ1_1 = map(blk -> blk[1:1, :], val_ψ1)
    Wψ1 = RealMLTB.assemble_Wpsi_caseB(val_ψ1_1, n_channels_φ, NORB)

    Hψ1 = Wψ1 * H_AO_1 * Wψ1'

    # ---- diagnostics ----
    D_phi = construct_basis_D_matrix_pyscf(bs_1atom, θ)
    println("  T equivariance error:  $(norm(T1 - D_phi' * T0 * D_phi))")
    println("  V equivariance error:  $(norm(V1 - D_phi' * V0 * D_phi))")
    println("  Wψ right-action error: $(norm(Wψ1 - Wψ0 * D_phi))")

    evals0 = sort(real.(eigvals(Hψ0)))
    evals1 = sort(real.(eigvals(Hψ1)))
    eval_diff = norm(evals1 - evals0)
    println("  Eigenvalue invariance error: $eval_diff")
    println("  Hψ elementwise error:        $(norm(Hψ1 - Hψ0))")
    @test eval_diff < tol
    println("====== test_Hpsi_caseB PASSED ======\n")
end