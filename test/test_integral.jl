using RealMLTB
using GaussianBasis
using Molecules
using LinearAlgebra
using Random
using Test

function main()
    atoms = [
        Molecules.Atom(14, 0.0, [0.0, 0.0, 0.0]),
        Molecules.Atom(14, 0.0, [2.35, 2.35, 2.35]),
    ]
    bs = BasisSet("def2-TZVP", atoms; spherical=true, lib=:acsint)

    println("=== Running both versions ===")
    T_dense,  V_dense,  S_dense  = compute_spherical_AO_integrals(bs)
    T_sparse, V_sparse, S_sparse = compute_spherical_AO_integrals_sparse(bs; cutoff=1e-12)

    # --- T and S should be identical between the two versions ---
    println("\n--- T / S comparison ---")
    println("max|ΔT| = ", maximum(abs, T_dense - T_sparse))
    println("max|ΔS| = ", maximum(abs, S_dense - S_sparse))
    @test maximum(abs, T_dense - T_sparse) < 1e-12
    @test maximum(abs, S_dense - S_sparse) < 1e-12

    # --- V comparison: expand the dense version into sparse format, then compare ---
    println("\n--- V comparison ---")

    N = size(T_dense, 1)
    cutoff = 1e-12

    # Flatten the dense V into (indices, vals) using the same cutoff
    dense_indices = NTuple{3,Int}[]
    dense_vals    = Float64[]
    for c in 1:N, b in 1:N, a in 1:N
        v = V_dense[a, b, c]
        if abs(v) > cutoff
            push!(dense_indices, (a, b, c))
            push!(dense_vals, v)
        end
    end

    sparse_indices, sparse_vals = V_sparse

    println("Dense  nnz: ", length(dense_vals))
    println("Sparse nnz: ", length(sparse_vals))

    # Build lookup dicts for element-wise comparison
    sparse_dict = Dict(zip(sparse_indices, sparse_vals))
    dense_dict  = Dict(zip(dense_indices,  dense_vals))

    # Check every nonzero in the dense version against the sparse version
    max_err = 0.0
    missing_in_sparse = 0
    for (idx, v_dense) in dense_dict
        v_sparse = get(sparse_dict, idx, 0.0)
        err = abs(v_dense - v_sparse)
        if err > cutoff
            missing_in_sparse += 1
        end
        max_err = max(max_err, err)
    end

    # Check for any extra elements present in sparse but absent in dense
    extra_in_sparse = 0
    for (idx, _) in sparse_dict
        if !haskey(dense_dict, idx)
            extra_in_sparse += 1
        end
    end

    println("max|ΔV| (element-wise) = ", max_err)
    println("Elements in dense missing from sparse: ", missing_in_sparse)
    println("Extra elements in sparse not in dense: ", extra_in_sparse)
    @test max_err <= cutoff
    @test missing_in_sparse == 0
    @test extra_in_sparse == 0

    # --- Contraction equivalence: contract V against a random vector and compare ---
    println("\n--- Contraction equivalence ---")
    Random.seed!(42)
    w = randn(N)

    # Dense contraction: V[i,j,k] * w[k] -> M[i,j]
    M_dense = reshape(reshape(V_dense, N * N, N) * w, N, N)

    # Sparse contraction: accumulate manually from the stored triplets
    M_sparse = zeros(N, N)
    for (n, (i, j, k)) in enumerate(sparse_indices)
        M_sparse[i, j] += sparse_vals[n] * w[k]
    end

    println("max|ΔM| (contracted matrix) = ", maximum(abs, M_dense - M_sparse))
    @test maximum(abs, M_dense - M_sparse) < 1e-10

    println("\n=== All comparisons done ===")
end

@testset "dense vs sparse spherical AO integrals" begin
    main()
end
