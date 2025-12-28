using Random
using LinearAlgebra
using StaticArrays
using Rotations
using WignerD
using EquivariantTensors.O3: Ctran

# Robustly extract val_scalar from various model return shapes.
# Handles:
#   val_scalar::AbstractMatrix
#   (val_scalar,)
#   NamedTuple with a single field (e.g. (ace = val_scalar,))
get_valscalar(val) = begin
    if val isa NamedTuple
        return getfield(val, fieldnames(typeof(val))[1])
    end
    if val isa Tuple && length(val) == 1
        return val[1]
    end
    return val
end

function rand_rot()
    Q = Matrix(qr(randn(3, 3)).Q)
    if det(Q) < 0
        Q[:, 1] *= -1
    end
    return SMatrix{3,3}(Q)
end

function rotate_graph(G, Q::SMatrix{3,3})
    new_edges = map(e -> merge(e, (; 𝐫 = Q * e.𝐫)), G.edge_data)
    return update_graph(G, new_edges)
end

function update_graph(old_G, new_edges)
    T = typeof(old_G)
    new_values = map(fieldnames(T)) do name
        if name == :edge_data
            return new_edges
        else
            return getfield(old_G, name)
        end
    end
    return T(new_values...)
end

function get_pyscf_permutation(L)
    if L == 0
        return [1]
    elseif L == 1
        return [3, 1, 2]
    elseif L == 2
        return [1, 2, 3, 4, 5]
    elseif L == 3
        return collect(1:7)
    else
        return collect(1:(2L+1))
    end
end

function construct_basis_D_matrix_pyscf(bs, θ)
    D_blocks = Matrix{Float64}[]
    for shell in bs.basis
        L = shell.l
        D_complex = transpose(wignerD(L, θ...))
        U = Ctran(L)
        D_std = real.(U * D_complex * U')
        P = get_pyscf_permutation(L)
        Dp = D_std[P, P]
        push!(D_blocks, Dp)
    end
    return cat(D_blocks..., dims=(1,2))
end

function construct_Dpsi_atom(nψ, θ)
    lmax = length(nψ) - 1
    blocks = Matrix{Float64}[]
    for L in 0:lmax
        nψL = nψ[L+1]
        nψL == 0 && continue

        D_complex = transpose(wignerD(L, θ...))
        U = Ctran(L)
        D_std = real.(U * D_complex * U')
        P = get_pyscf_permutation(L)
        Dp = D_std[P, P]

        for _ in 1:nψL
            push!(blocks, Dp)
        end
    end
    return cat(blocks..., dims=(1,2))
end

function construct_Dpsi_total(nψ, θ, NATOM::Int)
    D_atom = construct_Dpsi_atom(nψ, θ)
    Iat = Matrix{Float64}(I, NATOM, NATOM)
    return kron(Iat, D_atom)
end

# All equivariance / invariance tests

using LinearAlgebra
using StaticArrays
using Rotations
using WignerD
using Test
using EquivariantTensors.O3: Ctran

function test_equivariance(model, ps, st, G; tol=1e-5)
    println("\n====== Testing Equivariance (General L) ======")

    val_orig, _ = model(G, ps, st)

    θ = rand(3) * 2π
    Q = RotZYZ(θ...)
    Q_mat = SMatrix{3,3}(Q)

    G_rot = rotate_graph(G, Q_mat)
    val_rot, _ = model(G_rot, ps, st)

    for (i, v_orig_block) in enumerate(val_orig)
        L = i - 1
        v_rot_block = val_rot[i]

        println("--- Testing L = $L ---")

        if L == 0
            diff = norm(v_rot_block - v_orig_block)
            println("  L=0 Error: $(diff)")
            @test diff < tol
        else
            D_complex = transpose(wignerD(L, θ...))
            U = Ctran(L)
            D_real = real.(U * D_complex * U')

            val_expected = map(v_orig_block) do v
                SVector{2*L+1}(D_real * Vector(v))
            end

            diff = norm(v_rot_block - val_expected)
            println("  L=$L Error: $(diff)")
            @test diff < tol
        end
    end

    println("====== Test Passed ======\n")
end

function rotate_basis_set(bs_orig::BasisSet, Q)
    new_atoms = map(bs_orig.atoms) do atom
        new_xyz = Q * atom.xyz
        Molecules.Atom(atom.Z, atom.mass, new_xyz)
    end
    return BasisSet(bs_orig.name, new_atoms; spherical=false, lib=:acsint)
end

function test_V_pair_equivariance(modelWv, psWv, stWv, G, bs_cart_orig; tol=1e-5)
    println("\n====== Testing V_pair Matrix Equivariance ======")

    val_orig, _ = modelWv(G, psWv, stWv)
    Wv_orig = RealMLTB.assemble_Wv(val_orig)

    _, V_pair_raw, _ = compute_spherical_AO_integrals(bs_cart_orig)
    nbas = size(V_pair_raw, 1)
    M_orig = reshape(V_pair_raw, nbas^2, nbas)
    V_pair_orig = reshape(M_orig * Wv_orig, nbas, nbas)

    θ = rand(3) * 2π
    Q = RotZYZ(θ...)
    Q_mat = SMatrix{3,3}(Q)

    G_rot = rotate_graph(G, Q_mat)
    bs_cart_rot = rotate_basis_set(bs_cart_orig, Q_mat)

    val_rot, _ = modelWv(G_rot, psWv, stWv)
    Wv_rot = RealMLTB.assemble_Wv(val_rot)

    _, V_pair_raw_rot, _ = compute_spherical_AO_integrals(bs_cart_rot)
    M_rot = reshape(V_pair_raw_rot, nbas^2, nbas)
    V_pair_calc = reshape(M_rot * Wv_rot, nbas, nbas)

    D_total = construct_basis_D_matrix_pyscf(bs_cart_orig, θ)
    V_pair_expected = transpose(D_total) * V_pair_orig * D_total

    diff = norm(V_pair_calc - V_pair_expected)
    println("  Matrix Error: $(diff)")
    @test diff < tol
end

function build_V_AO_from_Wv(modelWv, psWv, stWv, G, bs_cart)
    valV, _ = modelWv(G, psWv, stWv)
    Wv = RealMLTB.assemble_Wv(valV)

    _, V_raw, _ = compute_spherical_AO_integrals(bs_cart)
    nbas = size(V_raw, 1)
    M = reshape(V_raw, nbas^2, nbas)
    return reshape(M * Wv, nbas, nbas)
end

function test_Wpsi_caseB(modelWpsi, psWpsi, stWpsi, G, bs_cart_orig, n_channels, NORB; tol=1e-5)
    println("\n====== Testing Wpsi Case-B (right action) ======")

    val0, _ = modelWpsi(G, psWpsi, stWpsi)
    W0 = RealMLTB.assemble_Wpsi_caseB(val0, n_channels, NORB)

    θ = rand(3) * 2π
    Q = RotZYZ(θ...)
    Q_mat = SMatrix{3,3}(Q)
    G_rot = rotate_graph(G, Q_mat)

    val1, _ = modelWpsi(G_rot, psWpsi, stWpsi)
    W1 = RealMLTB.assemble_Wpsi_caseB(val1, n_channels, NORB)

    D_total = construct_basis_D_matrix_pyscf(bs_cart_orig, θ)

    W_expected = W0 * D_total
    diff = norm(W1 - W_expected)
    println("  Wpsi error: $(diff)")
    @test diff < tol
end

function test_Hpsi_caseB(modelWv, psWv, stWv,
                         modelWpsi, psWpsi, stWpsi,
                         G, bs_cart_orig, n_channels, NORB; tol=1e-5)

    println("\n====== Testing H^psi elementwise invariance (Case B) ======")

    valψ0, _ = modelWpsi(G, psWpsi, stWpsi)
    Wpsi0 = RealMLTB.assemble_Wpsi_caseB(valψ0, n_channels, NORB)

    T0, _, _ = compute_spherical_AO_integrals(bs_cart_orig)
    V0 = build_V_AO_from_Wv(modelWv, psWv, stWv, G, bs_cart_orig)
    H_AO_0 = T0 + V0
    Hψ0 = Wpsi0 * H_AO_0 * transpose(Wpsi0)

    θ = rand(3) * 2π
    Q = RotZYZ(θ...)
    Q_mat = SMatrix{3,3}(Q)

    G_rot = rotate_graph(G, Q_mat)
    bs_cart_rot = rotate_basis_set(bs_cart_orig, Q_mat)

    valψ1, _ = modelWpsi(G_rot, psWpsi, stWpsi)
    Wpsi1 = RealMLTB.assemble_Wpsi_caseB(valψ1, n_channels, NORB)

    T1, _, _ = compute_spherical_AO_integrals(bs_cart_rot)
    V1 = build_V_AO_from_Wv(modelWv, psWv, stWv, G_rot, bs_cart_rot)
    H_AO_1 = T1 + V1
    Hψ1 = Wpsi1 * H_AO_1 * transpose(Wpsi1)

    diff = norm(Hψ1 - Hψ0)
    println("  H^psi error: $(diff)")
    @test diff < tol
end

function test_Wpsi_caseA(modelWpsi, psWpsi, stWpsi, G, bs_cart_orig, n_channels_φ, nψ; tol=1e-5)
    println("\n====== Testing Wpsi Case-A (two-sided) ======")

    val0, _ = modelWpsi(G, psWpsi, stWpsi)
    val0s = get_valscalar(val0)
    NATOM = size(val0s, 1)
    W0 = RealMLTB.assemble_Wpsi_caseA(val0s, n_channels_φ, nψ)

    θ = rand(3) * 2π
    Q = RotZYZ(θ...)
    Q_mat = SMatrix{3,3}(Q)
    G_rot = rotate_graph(G, Q_mat)

    val1, _ = modelWpsi(G_rot, psWpsi, stWpsi)
    val1s = get_valscalar(val1)
    W1 = RealMLTB.assemble_Wpsi_caseA(val1s, n_channels_φ, nψ)

    Dφ = construct_basis_D_matrix_pyscf(bs_cart_orig, θ)
    Dψ = construct_Dpsi_total(nψ, θ, NATOM)

    W_expected = transpose(Dψ) * W0 * Dφ

    diff = norm(W1 - W_expected)
    println("  Wpsi error: $(diff)")
    @test diff < tol
end

function test_Hpsi_caseA(modelWv, psWv, stWv,
                         modelWpsi, psWpsi, stWpsi,
                         G, bs_cart_orig, n_channels_φ, nψ; tol=1e-5)

    println("\n====== Testing H^psi similarity (Case A) ======")

    T0, _, _ = compute_spherical_AO_integrals(bs_cart_orig)

    valψ0, _ = modelWpsi(G, psWpsi, stWpsi)
    valψ0s = get_valscalar(valψ0)
    NATOM = size(valψ0s, 1)
    Wψ0 = RealMLTB.assemble_Wpsi_caseA(valψ0s, n_channels_φ, nψ)

    V0 = build_V_AO_from_Wv(modelWv, psWv, stWv, G, bs_cart_orig)
    H_AO_0 = T0 + V0
    Hψ0 = Wψ0 * H_AO_0 * transpose(Wψ0)

    θ = rand(3) * 2π
    Q = RotZYZ(θ...)
    Q_mat = SMatrix{3,3}(Q)

    G_rot = rotate_graph(G, Q_mat)
    bs_cart_rot = rotate_basis_set(bs_cart_orig, Q_mat)

    T1, _, _ = compute_spherical_AO_integrals(bs_cart_rot)

    valψ1, _ = modelWpsi(G_rot, psWpsi, stWpsi)
    valψ1s = get_valscalar(valψ1)
    Wψ1 = RealMLTB.assemble_Wpsi_caseA(valψ1s, n_channels_φ, nψ)

    V1 = build_V_AO_from_Wv(modelWv, psWv, stWv, G_rot, bs_cart_rot)
    H_AO_1 = T1 + V1
    Hψ1 = Wψ1 * H_AO_1 * transpose(Wψ1)

    Dψ = construct_Dpsi_total(nψ, θ, NATOM)
    H_expected = transpose(Dψ) * Hψ0 * Dψ

    diff = norm(Hψ1 - H_expected)
    println("  Hψ similarity error: $(diff)")
    @test diff < tol
end
