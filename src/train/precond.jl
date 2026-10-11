# ---------------------------------------------------------------------------
# Block preconditioning for the band fits.
#
# L-BFGS and BFGS are not affine invariant: the initial inverse Hessian is
# isotropic (gamma*I), so the coordinates the parameters happen to be written
# in decide how fast the optimiser moves.  Fixing those coordinates is worth
# a lot: on the Cu fits a preconditioned arm reached 151.480 meV by step 30
# where the unpreconditioned one needed hundreds of steps, and both converged
# to the same floor.  So this buys optimisation efficiency, not a better
# model, and should not be reported as an accuracy improvement.
#
# The metric.  The reference implementation uses
#     P_v = (G_v + tau_v lambda_max I)^{-1/2},  G_v = sum_g Dv_g' Dv_g
# the Gram of the LINEAR map from potential parameters to the potential, and
# Jacobi scaling on the Wb block.  Here the parameters enter nonlinearly, so
# the same object is the Gauss-Newton Gram of the band map,
#
#     G = J' W J,     J[(k,n), p] = d lambda_kn / d theta_p
#
# summed over structures and k with the SAME k and band weights the loss uses,
# so the metric matches the objective it preconditions.  It depends on the
# training geometries but not on the targets, exactly as G_v does.
#
# The regulariser is not cosmetic: raw G^{-1/2} amplifies the null directions
# without bound, measured at 1.2e-3 against a 1e-6 gate.  tau caps the
# amplification at tau^{-1/2}.
#
# Three gates, run by `precond_report`:
#   P1  ||P (Pinv theta0) - theta0|| / ||theta0||   exact change of variables
#   P2  ||P Pinv - I||_F                            invertibility
#   P3  finite differences against d/dt f(P (y0 + t d))
# ---------------------------------------------------------------------------

using Printf

export band_gram, build_preconditioner, precond_report

struct BandPreconditioner{TP, TI}
    P::TP           # theta = P y
    Pinv::TI        # y = Pinv theta
    mode::Symbol
    tau::Float64
    lam::Vector{Float64}    # spectrum of G, for reporting
end

Base.size(pc::BandPreconditioner) = size(pc.P)

"""
    band_gram(Glist, λ_ref_list, model, ps, st; train_part, bands, ref_bands,
              kweights_list, band_weights)

Gauss-Newton Gram  G = J' W J  of the band map, J[(k,n),p] = dλ_kn/dθ_p, over
every structure and k point, with the loss's own weights.  `λ_ref_list` is used
only for the band window, never for its values.

One reverse-mode pass per residual row, so the cost is (#rows) gradients.  That
is cheap at the sizes this branch runs at (np = 40..250), and the matrix is
np x np.
"""
function band_gram(Glist, λ_ref_list, model, ps, st;
                   train_part::Symbol = :all,
                   bands = nothing,
                   ref_bands = nothing,
                   kweights_list = nothing,
                   band_weights = nothing)

    θ0, _, _ = _make_objective_over_part(p -> 0.0, ps, train_part)
    part0, rebuild = _select_trainable_part(ps, train_part)
    θ_raw, re = destructure(part0)
    Tp = eltype(θ_raw)
    np = length(θ0)

    # θ -> the stacked vector of windowed eigenvalues, and the matching weights
    function λvec(θ)
        ps_cur = rebuild(re(Tp.(θ)))
        out = Float64[]
        for i in eachindex(Glist)
            λ = model(Glist[i], ps_cur, st)[1].λ
            for ik in eachindex(λ)
                w = bands === nothing ? real.(λ[ik]) : real.(λ[ik][bands])
                out = vcat(out, w)
            end
        end
        return out
    end

    roww = Float64[]
    for i in eachindex(Glist)
        λ = model(Glist[i], ps, st)[1].λ
        nb = bands === nothing ? length(first(λ)) : length(bands)
        b  = band_weights === nothing ? ones(nb) : Float64.(band_weights)
        kw = kweights_list === nothing ? ones(length(λ)) : Float64.(kweights_list[i])
        for ik in eachindex(λ)
            append!(roww, kw[ik] .* b)
        end
    end

    nrow = length(roww)
    J = zeros(nrow, np)
    for r in 1:nrow
        g = Zygote.gradient(θ -> λvec(θ)[r], θ0)[1]
        g === nothing || (J[r, :] = Float64.(g))
    end

    G = J' * (roww .* J)
    return Symmetric((G + G') / 2), J, roww
end

"""
    build_preconditioner(G; tau = 1e-6, mode = :full)

`:full` whitens with (G + tau*lambda_max I)^{-1/2} through an eigendecomposition.
`:jacobi` keeps the diagonal only, 1/sqrt(d_p + tau*max d), which is what the
other pipeline uses for its 12646-dimensional Wb block; prefer it above a few
thousand parameters.
"""
function build_preconditioner(G::AbstractMatrix; tau::Float64 = 1e-6,
                              mode::Symbol = :full,
                              normalize::Bool = false)
    mode in (:full, :jacobi) || error("mode must be :full or :jacobi, got $mode")

    # `normalize` divides P by max(scal) so no direction is stretched relative
    # to the original coordinates.  It is off by default: measured on the real
    # model it was worse, 0.0621 Ha against 0.0326 at the same 60-iteration
    # budget.  Note that it cannot change the ratio of the scalings, only the
    # overall factor, and that factor still matters because g_tol applies to
    # the PRECONDITIONED gradient, whose scale differs by ~171x here -- so the
    # two runs do not stop on the same criterion.  Any A/B here needs a
    # scale-free stopping rule first.
    if mode === :jacobi
        d = max.(diag(G), 0.0)
        t = tau * max(maximum(d), eps())
        p = 1 ./ sqrt.(d .+ t)
        normalize && (p = p ./ maximum(p))
        return BandPreconditioner(Diagonal(p), Diagonal(1 ./ p), :jacobi, tau, d)
    end

    F = eigen(Symmetric(Matrix(G)))
    lam = F.values
    t = tau * max(maximum(lam), eps())
    scal = 1 ./ sqrt.(max.(lam, 0.0) .+ t)
    normalize && (scal = scal ./ maximum(scal))
    P    = F.vectors * Diagonal(scal) * F.vectors'
    Pinv = F.vectors * Diagonal(1 ./ scal) * F.vectors'
    return BandPreconditioner((P + P') / 2, (Pinv + Pinv') / 2, :full, tau, lam)
end

"""
    precond_report(pc, θ0, f) -> NamedTuple

Run the three gates and print them.  `f` is the objective in θ coordinates.
Throws if P is not invertible to 1e-8 or if the directional-derivative chain
does not check out, because in either case the experiment is no longer "only
the coordinates changed".
"""
function precond_report(pc::BandPreconditioner, θ0::AbstractVector, f;
                        precision::Type = Float64,
                        steps = nothing,
                        tol = nothing)
    P1 = norm(pc.P * (pc.Pinv * θ0) - θ0) / max(norm(θ0), eps())
    P2 = norm(pc.P * pc.Pinv - I)

    # Conditioning has to be read on the subspace the Gram can actually see.
    # Directions with lambda below the tau floor are null to the metric;
    # whitening cannot condition them and tau's whole job is to stop them being
    # amplified.  Reporting max/min over ALL eigenvalues after whitening is
    # meaningless -- it divides by a number that is zero by construction.
    lam = pc.lam
    tv  = pc.tau * max(maximum(lam), eps())
    keep = lam .> tv
    n_eff = count(keep)
    cond_before = n_eff == 0 ? Inf :
        sqrt(maximum(lam[keep]) / max(minimum(lam[keep]), eps()))
    wh = lam[keep] ./ (lam[keep] .+ tv)          # spectrum of P' G P on that subspace
    cond_after = n_eff == 0 ? Inf : maximum(wh) / max(minimum(wh), eps())

    @printf("PRECONDITIONER  mode=%s tau=%.0e  np=%d\n", pc.mode, pc.tau, size(pc.P, 1))
    @printf("  lambda(G): max %.4e  min %.4e   %d of %d above the tau floor %.3e\n",
            maximum(lam), minimum(lam), n_eff, length(lam), tv)
    @printf("  sqrt-cond on that subspace: %.3e -> %.3e\n", cond_before, cond_after)
    if n_eff < length(lam)
        @printf("  %d of %d direction(s) are NULL to the Gram.\n",
                length(lam) - n_eff, length(lam))
        @printf("  A Gauss-Newton Gram has rank <= the number of residual rows, so\n")
        @printf("  np > (structures x k x bands) guarantees a null space no tau can\n")
        @printf("  repair.  Preconditioning directions the metric cannot see does not\n")
        @printf("  help and, unnormalised, actively hurts.  Add data rows or switch to\n")
        @printf("  a metric with better rank (the parameter -> Wv Gram).\n")
    end
    @printf("  GATE_P1  ||P(Pinv th0) - th0||/||th0|| = %.3e\n", P1)
    @printf("  GATE_P2  ||P Pinv - I||_F              = %.3e\n", P2)
    P2 < 1e-8 || error("P is not invertible to tolerance: ||P Pinv - I||_F = $P2")

    # GATE_P3's step sizes and tolerance have to match the objective's NOISE
    # FLOOR, not the ideal truncation error.  The objective is evaluated at the
    # parameters' precision (_make_objective_over_part rebuilds with
    # Tparam.(x)), and the ACE weights are Float32, so f carries ~1e-7 relative
    # quantisation noise.  A central difference then has error ~noise/eps, which
    # is 1e-3 at eps=1e-4 and gets WORSE as eps shrinks -- the first run of this
    # gate on the real model measured exactly that (1.14e-3, 1.66e-2, 5.49e-2
    # for 1e-4, 1e-5, 1e-6) and the gate failed, correctly, on round-off rather
    # than on a wrong chain rule.  The optimal step for a noisy function is
    # ~noise^(1/3) with achievable error ~noise^(2/3).
    u = eps(real(precision))
    if steps === nothing
        steps = precision === Float64 ? (1e-4, 1e-5, 1e-6) :
                                        (cbrt(u), 3cbrt(u), 10cbrt(u))
    end
    tol === nothing && (tol = precision === Float64 ? 1e-4 : 30 * u^(2/3))

    y0 = pc.Pinv * θ0
    F  = y -> f(pc.P * y)
    gy = Zygote.gradient(F, y0)[1]
    d  = gy ./ max(norm(gy), eps())
    best = Inf
    for ε in steps
        fd  = (F(y0 + ε .* d) - F(y0 - ε .* d)) / 2ε
        ana = dot(d, gy)
        rel = abs(fd - ana) / max(abs(ana), eps())
        @printf("  GATE_P3  eps=%.2e  FD=%+.8e  d'gy=%+.8e  rel=%.3e\n", ε, fd, ana, rel)
        best = min(best, rel)
    end
    best < tol || error("GATE_P3 failed: best relative error $best against a " *
        "tolerance of $tol for $(precision) parameters.  Either the " *
        "preconditioned gradient is not the chain rule of the original, or the " *
        "step sizes do not match the objective's noise floor -- pass steps=/tol=.")
    @printf("  GATE_P3  PASS  best rel = %.3e  (tol %.3e for %s parameters)\n",
            best, tol, precision)

    return (; P1, P2, P3 = best, cond_before, cond_after, n_eff, tol)
end
