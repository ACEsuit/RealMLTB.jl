using RealMLTB
using Test
using Random
using LinearAlgebra

# ---------------------------------------------------------------------------
# Gauge decomposition of a band error.
#
# res[k][n] = λ_pred[k][n] - λ_ref[k][n],  c_k = -mean_n res[k][n]
#
#   L_abs^2    == L_global^2 + c_global^2
#   L_global^2 == L_perk^2   + L_disp
#   L_perk <= L_global <= L_abs
# ---------------------------------------------------------------------------

@testset "band gauge decomposition" begin
    Random.seed!(7)
    nk, nb = 11, 6
    λ_ref = [sort(randn(nb)) for _ in 1:nk]

    cg   = 0.37                      # planted rigid shift
    offs = 0.11 .* randn(nk)         # planted per-k offsets
    λ_pred = [λ_ref[k] .+ cg .+ offs[k] .+ 0.02 .* randn(nb) for k in 1:nk]

    w = rand(nk) .+ 0.2

    for kw in (nothing, w)
        r = RealMLTB.band_gauge_report(λ_pred, λ_ref; keep_bands = nb, kweights = kw)

        # the two identities (also asserted inside band_gauge_report)
        @test isapprox(r.L_abs^2, r.L_global^2 + r.c_global^2; rtol = 1e-10)
        @test isapprox(r.L_global^2, r.L_perk^2 + r.L_disp;    rtol = 1e-10)
        @test r.misgauge ≈ r.c_global^2

        # ordering
        @test r.L_perk <= r.L_global + 1e-12
        @test r.L_global <= r.L_abs + 1e-12
    end

    # a rigid shift of the whole band structure is invisible to :global
    λ_rigid = [λ_ref[k] .+ 0.5 for k in 1:nk]
    r = RealMLTB.band_gauge_report(λ_rigid, λ_ref; keep_bands = nb)
    @test r.L_abs ≈ 0.5
    @test r.L_global < 1e-12
    @test r.L_perk  < 1e-12
    @test r.L_disp  < 1e-24
    @test r.c_global ≈ -0.5

    # a pure k-dependent offset is invisible to :perk and shows up entirely
    # as gauge dispersion -- this is why :perk must never be quoted alone
    λ_offs = [λ_ref[k] .+ offs[k] for k in 1:nk]
    r = RealMLTB.band_gauge_report(λ_offs, λ_ref; keep_bands = nb)
    @test r.L_perk < 1e-12
    @test isapprox(r.L_global^2, r.L_disp; rtol = 1e-12)
end

@testset "gauge helpers" begin
    res = [[1.0, 2.0], [3.0, 4.0]]
    w   = [1.0, 1.0]

    # :absolute is the identity
    @test RealMLTB._apply_gauge(res, w, :absolute) == res

    # :global removes the single weighted mean (2.5)
    g = RealMLTB._apply_gauge(res, w, :global)
    @test RealMLTB._global_shift(res, w) ≈ -2.5
    @test g[1] ≈ [-1.5, -0.5]
    @test g[2] ≈ [0.5, 1.5]

    # :perk removes one mean per k
    p = RealMLTB._apply_gauge(res, w, :perk)
    @test p[1] ≈ [-0.5, 0.5]
    @test p[2] ≈ [-0.5, 0.5]

    # weights enter both the shift and the norm
    @test RealMLTB._global_shift(res, [3.0, 1.0]) ≈ -(3*3 + 1*7)/8
    @test RealMLTB._weighted_rmse(res, [1.0, 1.0]) ≈ sqrt((1+4+9+16)/4)

    @test_throws ErrorException RealMLTB._apply_gauge(res, w, :nonsense)
end
