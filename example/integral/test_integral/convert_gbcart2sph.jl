# Verifies the correctness of integral transformations (Cartesian to Spherical) 
# and normalization scaling between GaussianBasis.jl and PySCF using 1-electron integrals (S, T).

using PyCall
using LinearAlgebra
using GaussianBasis
using Molecules
using StaticArrays
using Test

include("pyscf.jl")

# --- Setup PySCF Environment ---
pyscf = pyimport("pyscf")

mol_py = pyscf.M(atom="Si 0 0 0", basis="def2-svp", cart=false)
mol_py_cart = pyscf.M(atom="Si 0 0 0", basis="def2-svp", cart=true)

# Get PySCF Standard Data (Target)
Spy_sph = mol_py.intor("int1e_ovlp_sph")
Tpy_sph = mol_py.intor("int1e_kin")
Spy_cart = mol_py.intor("int1e_ovlp_cart")

# Get Transformation Matrix C
C = mol_py.cart2sph_coeff()

# --- Setup GaussianBasis Environment ---
atoms = [Molecules.Atom(14, 0.0, SVector{3}([0.0, 0.0, 0.0]))]

# Note: Cartesian must use `:acsint` to avoid the libcint d-shell bug
bs_sph = BasisSet("def2-svp", atoms; spherical=true, lib=:libcint)
bs_cart = BasisSet("def2-svp", atoms; spherical=false, lib=:acsint)

println("\n=== GaussianBasis Setup ===")
# Calculate GaussianBasis Raw Data
S_gb_sph = overlap(bs_sph)
T_gb_sph = kinetic(bs_sph)
S_gb_cart = overlap(bs_cart)
T_gb_cart = kinetic(bs_cart)

# --- Run Tests ---
println("\n=== Running Tests ===")

# Step A: Compute Normalization/Scaling Factors
# d_s: Scaling factors for Spherical basis (GB -> PySCF)
# d_c: Scaling factors for Cartesian basis (GB -> PySCF)
d_s = compute_scaling_factors(Spy_sph, S_gb_sph)
d_c = compute_scaling_factors(Spy_cart, S_gb_cart)

Ds = Diagonal(d_s)
Dc = Diagonal(d_c)

@testset "Normalization Checks" begin
    # Verify if normalization is correct (GB * D^2 == PySCF)
    S_gb2py_sph = Ds * S_gb_sph * Ds
    @test isapprox(S_gb2py_sph, Spy_sph, atol=1e-12)
    println("Spherical Norm Check Error: ", norm(S_gb2py_sph - Spy_sph))

    S_gb2py_cart = Dc * S_gb_cart * Dc
    @test isapprox(S_gb2py_cart, Spy_cart, atol=1e-12)
    println("Cartesian Norm Check Error: ", norm(S_gb2py_cart - Spy_cart))
end

@testset "Transformation Checks (Manual)" begin
    # Verify PySCF Internal Consistency: C' * S_cart_norm * C == S_sph_norm
    Spy_sph_recon = C' * Spy_cart * C
    @test isapprox(Spy_sph_recon, Spy_sph, atol=1e-12)
    println("PySCF Internal Consistency Error: ", norm(Spy_sph_recon - Spy_sph))
end

@testset "Full Chain Conversion (Cart -> Sph)" begin
    println("\n--- Testing Full Conversion Function ---")
    
    # Test 1: Overlap Matrix S
    # Goal: Derive S_gb_sph from S_gb_cart
    S_derived = convert_gb_cart_to_sph(S_gb_cart, C, d_c, d_s)
    
    diff_S = norm(S_derived - S_gb_sph)
    println("Overlap Matrix (S) Conversion Error: $diff_S")
    @test diff_S < 1e-12
    
    # Test 2: Kinetic Matrix T
    # Goal: Derive T_gb_sph from T_gb_cart
    T_derived = convert_gb_cart_to_sph(T_gb_cart, C, d_c, d_s)
    
    diff_T = norm(T_derived - T_gb_sph)
    println("Kinetic Matrix (T) Conversion Error: $diff_T")
    @test diff_T < 1e-12
end

println("\n=== All Tests Finished ===")
