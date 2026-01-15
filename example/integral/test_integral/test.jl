using RealMLTB
using Molecules, GaussianBasis
using AtomsBuilder, NeighbourLists
using EquivariantTensors
using Unitful: ustrip, Å
using LuxCore
using Random

include("data/generate_data.jl")
include("integral/integral.jl")
include("utils/build_graph.jl")
include("integral/test_integral/integral_pyscf.jl")

sys   = bulk(:Si)
X     = position(sys, :)
X_ang = [ustrip.(Å, r) for r in X]
mol_test = Molecules.Molecule([
    Molecules.Atom(14, 0.0, X_ang[1]), 
    Molecules.Atom(14, 0.0, X_ang[2])
])
bs_input = BasisSet("def2-SVP", mol_test.atoms; spherical=true, lib=:acsint)

println("1. PySCF version...")
T_ref, V_ref, S_ref = compute_spherical_AO_integrals_pyscf(bs_input)

println("2. Julia version...")
T_pure, V_pure, S_pure = compute_spherical_AO_integrals(bs_input)

diff_S = norm(S_ref - S_pure)
diff_T = norm(T_ref - T_pure)
diff_V = norm(V_ref - V_pure)

println("S (Overlap) Error: $diff_S")
println("T (Kinetic) Error: $diff_T")
println("V (3c-Over) Error: $diff_V")

@testset "Pure Julia Implementation Verification" begin
    @test diff_S < 1e-12
    @test diff_T < 1e-12
    @test diff_V < 1e-12
end