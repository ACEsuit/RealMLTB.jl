# test.jl  -- run with:  julia --project test/test.jl

using RealMLTB
using GaussianBasis
using Molecules
using AtomsBuilder
using NeighbourLists
using EquivariantTensors
using Unitful
using UnitfulAtomic
using StaticArrays
using LinearAlgebra
using Random
using Test

include("rotation.jl")

Random.seed!(42)

rcut = 5.0u"Å"  

# ============================================================================
# System: 2-atom Si bulk
# ============================================================================
println("=== Setting up 2-atom Si bulk system ===")
sys   = bulk(:Si)
nlist = PairList(sys, rcut)
X     = position(sys, :)
X_ang = [ustrip.(u"Å", r) for r in X]

bs_cart = BasisSet(
    "def2-SVP",
    [Molecules.Atom(14, 0.0, X_ang[i]) for i in eachindex(X_ang)];
    spherical = false, lib = :acsint,
)
lmax, n_channels = RealMLTB.lmax_and_nchannels_per_atom(bs_cart)
println("  lmax=$lmax  n_channels=$n_channels")

klist = [SVector(0.0, 0.0, 0.0)]
println("=== Building ETGraph ===")
G = RealMLTB.nlist2graph(bs_cart, nlist, sys, klist)
println("  nnodes=$(EquivariantTensors.nnodes(G))  nedges=$(EquivariantTensors.nedges(G))")

# ============================================================================
# Build models
# ============================================================================
println("\n=== Building models ===")

modelWv, psWv, stWv = RealMLTB.build_ace_equivariant_model(n_channels, rcut)
println("  Wv model built.")

nψ = (1, 1, 0, 0, 0)
modelWpsiA, psWpsiA, stWpsiA =
    RealMLTB.build_ace_caseA_Wpsi_model(n_channels, nψ, rcut)
println("  Wpsi Case-A built.  nψ=$nψ")

NORB_B = 4
modelWpsiB, psWpsiB, stWpsiB =
    RealMLTB.build_ace_caseB_Wpsi_model(n_channels, NORB_B, rcut)
println("  Wpsi Case-B built.  NORB=$NORB_B")

# ============================================================================
# Run tests
# Hpsi tests use G.graph_data.T_on / V_on (on-site single-atom integrals)
# so AO index ordering matches Wpsi per-atom column layout exactly.
# ============================================================================
println("\n=== Running equivariance tests ===")

@testset "RealMLTB equivariance" begin

    @testset "Raw ACE equivariance (Wv)" begin
        test_equivariance(modelWv, psWv, stWv, G; tol = 1e-4)
    end

    @testset "V matrix equivariance" begin
        test_V_pair_equivariance(modelWv, psWv, stWv, G, bs_cart; tol = 1e-3)
    end

    @testset "Wpsi Case-A equivariance" begin
        test_Wpsi_caseA(modelWpsiA, psWpsiA, stWpsiA, G, bs_cart,
                        n_channels, nψ; tol = 1e-3)
    end

    @testset "Wpsi Case-B equivariance" begin
        test_Wpsi_caseB(modelWpsiB, psWpsiB, stWpsiB, G, bs_cart,
                        n_channels, NORB_B; tol = 1e-3)
    end

    # Hpsi end-to-end tests require the full construct_integral pipeline.
    # The naive slicing Wv[1:nbas] does not correctly extract per-atom
    # potential coefficients because assemble_Wv and V3c use different
    # normalisation conventions.  These tests are skipped until a proper
    # per-atom H_AO extraction is implemented via siteblock / construct_integral.
    @testset "Hpsi Case-A similarity invariance" begin
        @test_skip "Hpsi Case-A: skipped pending proper per-atom H_AO extraction"
    end

    @testset "Hpsi Case-B elementwise invariance" begin
        @test_skip "Hpsi Case-B: skipped pending proper per-atom H_AO extraction"
    end

end

println("\n=== All tests complete ===")