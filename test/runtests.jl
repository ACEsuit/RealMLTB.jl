using RealMLTB
using Test

# test_gradient.jl was removed.  It include()d test/data/generate_data.jl and
# test/data/graphene_ham.jl, neither of which exists in the repository, and it
# was written against a reference pipeline returning (bs, H_ref_list, G_list),
# i.e. one that fits the Hamiltonian matrix directly.  The surviving pipeline
# returns eigenvalues and subspaces, so reinstating the test means reinstating
# that data generator.

@testset "RealMLTB.jl" begin
    @testset "cart2sph" begin
        include("test_cart2sph.jl")
    end
    @testset "dense vs sparse integrals" begin
        include("test_integral.jl")
    end
    @testset "EPI quadrature reference" begin
        include("test_EPI.jl")
    end
    @testset "three-centre integrals" begin
        include("test_v3c.jl")
    end
    @testset "periodic potential sum" begin
        include("test_periodic_potential.jl")
    end
    @testset "CG-coupled reduction" begin
        include("test_wb_cg.jl")
    end
    @testset "cutoff criterion and overlap definiteness" begin
        include("test_cutoff.jl")
    end
    @testset "rotation end-to-end" begin
        include("test_rotation_e2e.jl")
    end
    @testset "RMSE_H" begin
        include("test_rmse_h.jl")
    end
    @testset "burg barrier" begin
        include("test_burg.jl")
    end
    @testset "band loss" begin
        include("test_band_loss.jl")
    end
    @testset "preconditioner and restarts" begin
        include("test_precond.jl")
    end
    @testset "gauge" begin
        include("test_gauge.jl")
    end
end
