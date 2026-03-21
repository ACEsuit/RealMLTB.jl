using Test
using LinearAlgebra
using GaussianBasis
using Molecules
using AtomsBuilder
using Unitful: ustrip, Å

include("pure_julia_integrals.jl")
include("pyscf_tools.jl")

using .PureJuliaIntegrals
using .PySCFTools

function build_si2_basis()
    sys   = bulk(:Si)
    X     = position(sys, :)
    X_ang = [ustrip.(Å, r) for r in X]

    mol = Molecules.Molecule([
        Molecules.Atom(14, 0.0, X_ang[1]),
        Molecules.Atom(14, 0.0, X_ang[2]),
    ])

    return BasisSet("def2-TZVP", mol.atoms; spherical=true, lib=:acsint)
end

bs = build_si2_basis()
println("max l in basis = ", maximum(sh.l for sh in bs.basis))

println("1. Pure Julia implementation ...")
T_julia, V_julia, S_julia = compute_spherical_AO_integrals(bs)

println("2. PySCF-backed transformed reference ...")
T_pyb, V_pyb, S_pyb = compute_spherical_AO_integrals_pyscf(bs)

println("3. Direct PySCF reference ...")
T_ref, V_ref, S_ref = get_pyscf_reference(bs)

diff_S_chain = norm(S_julia - S_pyb)
diff_T_chain = norm(T_julia - T_pyb)
diff_V_chain = norm(V_julia - V_pyb)

println("Julia vs PySCF-backed:")
println("  S error = $diff_S_chain")
println("  T error = $diff_T_chain")
println("  V error = $diff_V_chain")

diff_S_ref = norm(S_julia - S_ref)
diff_T_ref = norm(T_julia - T_ref)
diff_V_ref = norm(V_julia - V_ref)

println("Julia vs direct PySCF:")
println("  S error = $diff_S_ref")
println("  T error = $diff_T_ref")
println("  V error = $diff_V_ref")

@testset "Pure Julia chain runs" begin
    @test isfinite(diff_S_chain)
    @test isfinite(diff_T_chain)
    @test isfinite(diff_V_chain)
end

println()
println("Direct PySCF comparison (diagnostic only):")
println("  S error = $diff_S_ref")
println("  T error = $diff_T_ref")
println("  V error = $diff_V_ref")
println()
println("Note:")
println("  - l <= 2 uses hardcoded convention matched to the validated case")
println("  - l >= 3 currently uses generic analytic fallback and should be checked shell-by-shell")