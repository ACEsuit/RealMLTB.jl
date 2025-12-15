# Tests that both the ML model outputs and the constructed Hamiltonian matrices respect O(3) symmetry, 
# explicitly handling Wigner-D rotations and PySCF-specific basis ordering.

using Lux, Random
using Rotations, WignerD, LinearAlgebra, StaticArrays, Test
using EquivariantTensors.O3: Ctran

# --- Helper: Generate a random 3D Rotation Matrix ---
function rand_rot()
    # QR decomposition of random matrix gives a uniform random rotation
    Q = Matrix(qr(randn(3, 3)).Q)
    # Ensure determinant is +1 (proper rotation, not reflection)
    if det(Q) < 0
        Q[:, 1] *= -1
    end
    return SMatrix{3,3}(Q)
end

# --- Helper: Rotate the Input Graph ---
function rotate_graph(G, Q::SMatrix{3,3})
    new_edges = map(e -> merge(e, (; 𝐫 = Q * e.𝐫)), G.edge_data)
    G_rotate = update_graph(G, new_edges)
    return G_rotate
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

# --- The Main Equivariance Test Function ---
function test_equivariance(model, ps, st, G; tol=1e-5)
    println("\n====== Testing Equivariance (General L) ======")
    
    # Forward pass (Original)
    val_orig, _ = model(G, ps, st)
    
    # Generate Random Rotation
    θ = rand(3) * 2π      # Euler angles (α, β, γ)
    Q = RotZYZ(θ...)      # 3x3 Rotation Matrix (Active rotation)
    Q_mat = SMatrix{3,3}(Q)

    # Rotate Graph and Forward pass
    G_rot = rotate_graph(G, Q_mat)
    val_rot, _ = model(G_rot, ps, st)

    # Loop over each output to test equivariance
    # val_orig[1] -> L=0
    # val_orig[2] -> L=1
    # ...
    for (i, v_orig_block) in enumerate(val_orig)
        L = i - 1
        v_rot_block = val_rot[i]
        
        println("--- Testing L = $L ---")

        if L == 0
            # --- Scalar Case (Invariant) ---
            diff = norm(v_rot_block - v_orig_block)
            println("  L=0 Error: $(diff)")
            @test diff < tol
        else
            # --- Vector/Tensor Case (Equivariant) ---
            D_complex = transpose(wignerD(L, θ...))
            
            # (Complex -> Real Spherical Harmonics)
            U = Ctran(L) 
        
            # D_real = U * D_complex * U'
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

function construct_basis_D_matrix_pyscf(bs, θ)
    D_blocks = Matrix{Float64}[]
    
    for shell in bs.basis
        L = shell.l
        D_complex = transpose(wignerD(L, θ...))
    
        U = Ctran(L)
        D_std = real.(U * D_complex * U')
        P = get_pyscf_permutation(L)
        D_pyscf = D_std[P, P]
        
        push!(D_blocks, D_pyscf)
    end
    
    return cat(D_blocks..., dims=(1,2))
end

function get_pyscf_permutation(L)
    if L == 0
        return [1]
    elseif L == 1
        return [3, 1, 2]
    elseif L == 2
        return [1, 2, 3, 4, 5] 
    elseif L == 3
        return 1:7
    else
        return 1:(2L+1)
    end
end

function test_V_pair_equivariance(model, ps, st, G, bs_cart_orig; tol=1e-5)
    println("\n====== Testing V_pair Matrix Equivariance ======")
    
    # Computing Original V_pair
    val_orig, _ = model(G, ps, st)
    Wv_orig = RealMLTB.assemble_Wv(val_orig)
    
    _, V_pair_raw, _ = compute_spherical_AO_integrals(bs_cart_orig)
    nbas = size(V_pair_raw, 1)
    M_orig  = reshape(V_pair_raw, nbas^2, nbas)
    
    V_pair_orig = reshape(M_orig * Wv_orig, nbas, nbas)

    # Rotating System and Recomputing
    θ = rand(3) * 2π
    Q = RotZYZ(θ...)
    Q_mat = SMatrix{3,3}(Q)
    
    G_rot = rotate_graph(G, Q_mat)
    bs_cart_rot = rotate_basis_set(bs_cart_orig, Q_mat) 
    
    val_rot, _ = model(G_rot, ps, st)
    Wv_rot = RealMLTB.assemble_Wv(val_rot)

    _, V_pair_raw, _ = compute_spherical_AO_integrals(bs_cart_rot)
    nbas = size(V_pair_raw, 1)
    M_rot  = reshape(V_pair_raw, nbas^2, nbas)
    V_pair_calc = reshape(M_rot * Wv_rot, nbas, nbas)

    #  Verifying V_pair' == D^T * V_orig * D
    D_total = construct_basis_D_matrix_pyscf(bs_cart_orig, θ)
    
    #    V_expected = D^T * V_orig * D
    V_pair_expected = transpose(D_total) * V_pair_orig * D_total
    
    diff = norm(V_pair_calc - V_pair_expected)
    println("  Matrix Error: $(diff)")
    
    @test diff < tol
end
