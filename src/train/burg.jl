# ---------------------------------------------------------------------------
# Burg barrier on the reduced overlap.
#
# S_out = Wb S_AO Wb' loses positive definiteness as Wb approaches rank
# deficiency, and generate_λ_and_vecs then throws from cholesky(S) before any
# eigenvalue exists, so a loss's own isfinite guard cannot see it.
# _make_objective_over_part catches the throw and rejects the step; this
# penalty is what keeps the optimiser away from the boundary in the first
# place.
#
#     (fac/2) * sum_k ( tr G_k - logdet G_k - M )
#       = (fac/2) * sum_k sum_i ( g_i - log g_i - 1 )   >= 0
#
# zero only at G = I, divergent as any g_i -> 0.  Despite the name it is not a
# rank cutoff: it acts on the whole spectrum of S_out, so a merely
# ill-conditioned basis is already penalised before it becomes singular.
#
# The useful weight is not portable, because the barrier's gradient can sit
# several orders below the band term's at the start of a run and still
# dominate later.  Set it from
#     eta = |g_barrier| / |g_band|
# which `burg_report` computes, rather than from the raw value.
# ---------------------------------------------------------------------------

using Printf

export burg_penalty, burg_report, assembled_HS, overlap_report, check_overlap_pd

# ---------------------------------------------------------------------------
# assembled_HS
#
# H(k), S(k) as assembled, without the eigensolver: runs the model's first two
# layers only.  _final_layer calls generate_λ_and_vecs, which
# Cholesky-factorises S and throws PosDefException when S(k) is not positive
# definite, i.e. exactly in the case one wants to inspect.
# ---------------------------------------------------------------------------
function assembled_HS(model, G, ps, st)
    sub = Lux.Chain(layer_1 = model.layers.layer_1,
                    layer_2 = model.layers.layer_2)
    out, _ = sub(G, (layer_1 = ps.layer_1, layer_2 = ps.layer_2),
                    (layer_1 = st.layer_1, layer_2 = st.layer_2))
    return out[1], out[2]
end

# ---------------------------------------------------------------------------
# overlap_report
#
# min eig S(k) over the k grid, one row per structure:
# (structure, lmin, lmax, k_worst, ok).
#
# The assembled overlap can be indefinite independently of Wb.  The exact
# Bloch sum is a Gram matrix, hence positive definite at every k, but the
# windowed sum S_w(k) = Σ_R w(|R|) S_R exp(i k R) is only PSD when w is a
# positive-definite radial function on R³ (Bochner, plus the Schur product
# theorem), and fun_cut's smoothstep is not one.  A negative value here means
# the smooth cutoff is biting a neighbour shell where the overlap is still
# large; the remedy is to widen the pair-list cutoff until the tapered shell
# is negligible (max|V2| there of order 1e-2 is enough on Si, 2e-1 is not).
#
# This is not the failure mode burg_penalty guards.  That one is Wb losing row
# rank; here S is already damaged before Wb touches it, so no penalty on the
# parameters can reach it.
# ---------------------------------------------------------------------------
function overlap_report(model, Glist, ps, st; verbose::Bool = true)
    rows = map(enumerate(Glist)) do (idx, G)
        _, Slist = assembled_HS(model, G, ps, st)
        ls = [minimum(real.(eigvals(Hermitian(Matrix(S))))) for S in Slist]
        hs = [maximum(real.(eigvals(Hermitian(Matrix(S))))) for S in Slist]
        iw = argmin(ls)
        (structure = idx, lmin = ls[iw], lmax = maximum(hs),
         k_worst = iw, ok = ls[iw] > 0)
    end
    if verbose
        for r in rows
            @printf("structure %3d  min eig S(k) = %+.4e at k index %d  (max %.4e)  %s\n",
                    r.structure, r.lmin, r.k_worst, r.lmax,
                    r.ok ? "" : "*** NOT POSITIVE DEFINITE ***")
        end
        nb = count(!r.ok for r in rows)
        nb > 0 && @warn "assembled S(k) is indefinite for $nb of $(length(rows)) structures; generate_λ_and_vecs will throw PosDefException there"
    end
    return rows
end

"""
    burg_penalty(Sout; weight = 1.0)

`(weight/2) * mean_k ( tr G - logdet G - M )` over the k points, G = S_out[k].

Normalised by n_k and M so the number does not change meaning when either
changes.

Uses a Cholesky for the log-determinant, so it throws if S_out is ALREADY
indefinite.  That is deliberate: the barrier keeps a positive-definite S_out
away from the boundary, it cannot rescue one that has crossed it.  The
PosDefException catch in _make_objective_over_part handles the crossing.
"""
function burg_penalty(Sout; weight::Real = 1.0)
    nk = length(Sout)
    nk == 0 && return 0.0
    M = size(_kmat(Sout, 1), 1)

    total = sum(1:nk) do ik
        G = _kmat(Sout, ik)
        Gh = (G .+ G') ./ 2
        C = cholesky(Hermitian(Gh))
        logdetG = 2 * sum(log, real.(diag(C.U)))
        real(tr(Gh)) - logdetG - M
    end

    return weight * total / (2 * nk * M)
end

"""
    burg_report(model, Glist, λ_ref_list, ps, st; weight, train_part, kwargs...)

The gradient ratio that decides the weight:

    eta = |grad of the barrier| / |grad of the band term|

plus the two values and the extreme eigenvalues of S_out.  `kwargs` are passed
to `RMSE_bands` so the band term is the one you are actually training on.

The other pipeline measured eta = 3.6e-3 late in training and 5.3e-7 at z = 0
with its default weight; pick a weight that puts eta where you want it and
re-measure, rather than copying a number across models.
"""
function burg_report(model, Glist, λ_ref_list, ps, st;
                     weight::Real = 1.0,
                     train_part::Symbol = :all,
                     kwargs...)

    band(p) = RMSE_bands(Glist, λ_ref_list, model, p, st; kwargs...)
    barr(p) = sum(burg_penalty(model(G, p, st)[1].Sout; weight = weight)
                  for G in Glist) / length(Glist)

    _, fb, _ = _make_objective_over_part(band, ps, train_part)
    θ0, fr, _ = _make_objective_over_part(barr, ps, train_part)

    gb = Zygote.gradient(fb, θ0)[1]
    gr = Zygote.gradient(fr, θ0)[1]
    nb = gb === nothing ? 0.0 : norm(gb)
    nr = gr === nothing ? 0.0 : norm(gr)

    λmin, λmax = Inf, -Inf
    for G in Glist
        for S in model(G, ps, st)[1].Sout
            ev = eigvals(Hermitian((S .+ S') ./ 2))
            λmin = min(λmin, minimum(real.(ev)))
            λmax = max(λmax, maximum(real.(ev)))
        end
    end

    return (; band = fb(θ0), barrier = fr(θ0),
              grad_band = nb, grad_barrier = nr,
              eta = nb == 0 ? Inf : nr / nb,
              S_out_eigmin = λmin, S_out_eigmax = λmax,
              weight)
end


# ---------------------------------------------------------------------------
# check_overlap_pd
#
# Pre-flight check for the training entry points.  `mode` is :error
# (default), :warn or :off.
#
# :error rather than :warn because an indefinite S(k) does not degrade a fit,
# it prevents one: every evaluation throws PosDefException,
# _make_objective_over_part turns that into Inf, and the optimiser then spends
# its whole budget on a constant objective.  That catch is the right behaviour
# for a line search stepping into a bad region, but here there is no good
# region to step back to.
# ---------------------------------------------------------------------------
function check_overlap_pd(model, Glist, ps, st;
                          mode::Symbol = :error,
                          verbose::Bool = true,
                          label::AbstractString = "")
    mode === :off && return nothing
    mode in (:error, :warn) || throw(ArgumentError(
        "check_overlap must be :error, :warn or :off, got :$mode"))

    Gs = Glist isa AbstractVector ? Glist : [Glist]
    isempty(Gs) && return nothing

    rows = try
        overlap_report(model, Gs, ps, st; verbose = false)
    catch e
        # a model whose first two layers are not (centre layer, assembly) can
        # not be inspected this way, and that is no reason to refuse to train
        @warn "$(label)could not inspect the assembled overlap; skipping the check" exception = e
        return nothing
    end

    return _overlap_verdict(rows, mode; verbose = verbose, label = label)
end

# split out from check_overlap_pd so it can be tested on synthetic rows,
# without a model that actually produces an indefinite overlap
function _overlap_verdict(rows, mode::Symbol;
                          verbose::Bool = true,
                          label::AbstractString = "")
    bad = filter(r -> !r.ok, rows)
    if isempty(bad)
        verbose && @printf("%soverlap check: min eig S(k) = %+.4e over %d structure(s), positive definite\n",
                           label, minimum(r.lmin for r in rows), length(rows))
        return rows
    end

    w = argmin([r.lmin for r in rows])
    msg = label * "the assembled overlap S(k) is NOT positive definite for " *
          "$(length(bad)) of $(length(rows)) structure(s); worst is structure " *
          "$(rows[w].structure) at k index $(rows[w].k_worst), min eig = " *
          "$(rows[w].lmin).  generate_λ_and_vecs Cholesky-factorises S, so it " *
          "will throw PosDefException at every evaluation: the objective is " *
          "Inf everywhere and the fit cannot start.  This is not a Wb problem " *
          "and the Burg barrier cannot reach it -- the damage is in S before " *
          "Wb touches it.  Widen the pair-list cutoff so the smooth cutoff " *
          "lands where the overlap is actually negligible (max|V2| at the " *
          "tapered shell ~1e-2 sufficed on Si, 2.4e-1 did not), or pass " *
          "check_overlap = :warn to proceed anyway."
    mode === :error ? error(msg) : @warn msg
    return rows
end