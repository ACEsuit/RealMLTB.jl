using LinearAlgebra
using StaticArrays
using ChainRulesCore
using Zygote
using Optimisers: destructure

export FermiFilter, RMSE, RMSE_H, band_gauge_report

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# CPU path: Hout/Sout are usually Vector{Matrix}.
@inline _kmat(xs::AbstractVector, ik::Integer) = xs[ik]

# GPU dense-assembly path: Hout/Sout are CuArray/AbstractArray with shape
# (dim, dim, nk).  Do not use xs[ik], because that is scalar linear indexing.
@inline function _kmat(xs::AbstractArray{T,3}, ik::Integer) where {T}
    return copy(@view xs[:, :, ik])
end

# Avoid Diagonal(...) * CuArray matrix products in RMSE_H.
# Julia's generic LinearAlgebra Diagonal multiplication can fall back to
# scalar indexing in reverse mode on GPU arrays.
#
# For a diagonal vector f and Hermitian Gram matrix S:
#     tr(D(f) * S * D(f) * S) = sum_{i,j} f_i f_j |S_ij|^2
#
# For two diagonal vectors fL, fR and matrix C:
#     tr(D(fL) * C * D(fR) * C') = sum_{i,j} fL_i fR_j |C_ij|^2
@inline function _diag_weighted_gram_sq(f, S)
    wf = reshape(f, length(f), 1) .* reshape(f, 1, length(f))
    return real(sum(wf .* abs2.(S)))
end

@inline function _diag_cross_weighted_gram(fL, fR, C)
    wf = reshape(fL, length(fL), 1) .* reshape(fR, 1, length(fR))
    return real(sum(wf .* abs2.(C)))
end

# ---------------------------------------------------------------------------
# Gauge handling for eigenvalue losses
#
# A band fit is determined only up to the absolute energy zero of the
# reference calculation (G = 0 convention, pseudopotential reference, ...).
# Three ways to take the difference λ_pred - λ_ref:
#
#   :absolute   no shift at all.
#   :global     ONE scalar for the whole structure, shared by every k point
#               and every band.  This is the "one global shift" judge.
#   :perk       a separate scalar for each k point.  DIAGNOSTIC ONLY: it
#               discards the k-to-k alignment, which is physical information,
#               and will report an error that no single Hamiltonian can
#               actually achieve.
#
# With  res[k][n] = λ_pred[k][n] - λ_ref[k][n],
#       c_k       = -mean_n res[k][n]            (per-k optimal shift)
#       c̄        = Σ_k (w_k/W) c_k              (W = Σ_k w_k n_k)
#
#       L_global^2 = L_perk^2 + L_disp ,
#       L_disp     = Σ_k (w_k n_k / W) (c_k - c̄)^2 .
#
# L_disp is the gauge dispersion: how much of the "one global shift" error is
# nothing but a k-dependent rigid offset.  Report it next to L_global, never
# report L_perk on its own.
# ---------------------------------------------------------------------------

const GAUGES = (:absolute, :global, :perk)

@inline _nband(r) = length(r)

"""
    _uniform_weights(res) -> Vector{Float64}

Default k weights: uniform, so that the weighted formulas below reduce
exactly to the plain (unweighted) RMSE.
"""
_uniform_weights(res) = ones(Float64, length(res))

"""
    _global_shift(res, w) -> c

The single scalar c minimising Σ_k w_k Σ_n (res[k][n] + c)^2, i.e. minus the
w-weighted mean of all residuals.  Differentiable (no mutation).
"""
function _global_shift(res, w)
    num = sum(w .* map(sum, res))
    den = sum(w .* map(_nband, res))
    return -num / den
end

"""
    _perk_shifts(res) -> Vector

c_k = -mean_n res[k][n].
"""
_perk_shifts(res) = map(r -> -sum(r) / _nband(r), res)

"""
    _apply_gauge(res, w, gauge) -> res'

Return the gauge-reduced residuals.  `:absolute` is the identity.
"""
function _apply_gauge(res, w, gauge::Symbol)
    gauge === :absolute && return res
    if gauge === :global
        c = _global_shift(res, w)
        return map(r -> r .+ c, res)
    elseif gauge === :perk
        return map(r -> r .- (sum(r) / _nband(r)), res)
    end
    error("unknown gauge $gauge; expected one of $GAUGES")
end

"""
    _weighted_rmse(res, w) -> sqrt( Σ_k w_k Σ_n res[k][n]^2 / Σ_k w_k n_k )
"""
function _weighted_rmse(res, w)
    sq = sum(w .* map(r -> sum(abs2, r), res))
    n  = sum(w .* map(_nband, res))
    return sqrt(sq / n)
end

# ---------------------------------------------------------------------------
# Eigenvalue RMSE
# ---------------------------------------------------------------------------

function RMSE(G, λ_ref, model, ps, st; keep_bands::Int = 4,
              gauge::Symbol = :global, kweights = nothing)
    result = model(G, ps, st)
    out = result[1]
    λ_pred = out.λ

    if any(v -> !all(isfinite, v), λ_pred)
        return eltype(first(λ_pred))(1e6)
    end

    @assert length(λ_pred) == length(λ_ref)

    res = [λ_pred[ik][1:keep_bands] .- λ_ref[ik][1:keep_bands]
           for ik in eachindex(λ_pred)]
    w   = isnothing(kweights) ? _uniform_weights(res) : collect(kweights)

    return _weighted_rmse(_apply_gauge(res, w, gauge), w)
end

# NOTE on the list version: the :global shift is fitted ONE PER STRUCTURE.
# Each reference calculation carries its own arbitrary energy zero, so a
# single shift shared across structures would mix a physical quantity
# (structure-to-structure level alignment) into the gauge.
function RMSE(Glist::AbstractVector, λ_ref_list::AbstractVector,
              model, ps, st; keep_bands::Int = 4,
              gauge::Symbol = :global, kweights_list = nothing)
    @assert length(Glist) == length(λ_ref_list)

    total_sq = 0.0
    total_n  = 0.0

    for i in eachindex(Glist)
        result = model(Glist[i], ps, st)
        out    = result[1]
        λ_pred = out.λ

        if any(v -> !all(isfinite, v), λ_pred)
            return eltype(first(λ_pred))(1e6)
        end

        @assert length(λ_pred) == length(λ_ref_list[i])

        res = [λ_pred[ik][1:keep_bands] .- λ_ref_list[i][ik][1:keep_bands]
               for ik in eachindex(λ_pred)]
        w   = isnothing(kweights_list) ? _uniform_weights(res) :
                                         collect(kweights_list[i])

        res_g     = _apply_gauge(res, w, gauge)
        total_sq += sum(w .* map(r -> sum(abs2, r), res_g))
        total_n  += sum(w .* map(_nband, res_g))
    end

    return sqrt(total_sq / total_n)
end

# ---------------------------------------------------------------------------
# Gauge decomposition of a band error (diagnostics / reporting)
# ---------------------------------------------------------------------------

"""
    band_gauge_report(λ_pred, λ_ref; keep_bands = 4, kweights = nothing)

Decompose the band error of ONE structure into its gauge components.

Returns a NamedTuple with

    L_abs     no shift                         (strictest)
    L_global  one global shift, all k and all bands share it  <- quote this
    L_perk    one shift per k point            (diagnostic only)
    L_disp    L_global^2 - L_perk^2, the gauge dispersion
    misgauge  L_abs^2 - L_global^2 == c_global^2
    c_global  the fitted global shift
    c_k       the per-k shifts

Exact identities, both asserted here:

    L_abs^2    == L_global^2 + c_global^2
    L_global^2 == L_perk^2   + L_disp

`L_perk` is never a defensible number on its own: it is the error of a
Hamiltonian that is allowed a different energy zero at every k point, which
no single Hamiltonian is.  Report `L_global` and `L_disp` together.
"""
function band_gauge_report(λ_pred, λ_ref; keep_bands::Int = 4, kweights = nothing)
    res = [λ_pred[ik][1:keep_bands] .- λ_ref[ik][1:keep_bands]
           for ik in eachindex(λ_pred)]
    w   = isnothing(kweights) ? _uniform_weights(res) : collect(kweights)

    W = sum(w .* map(_nband, res))

    c_global = _global_shift(res, w)
    c_k      = _perk_shifts(res)

    L_abs    = _weighted_rmse(res, w)
    L_global = _weighted_rmse(_apply_gauge(res, w, :global), w)
    L_perk   = _weighted_rmse(_apply_gauge(res, w, :perk), w)

    L_disp = sum(w .* map(_nband, res) .* (c_k .- c_global) .^ 2) / W

    @assert isapprox(L_abs^2, L_global^2 + c_global^2; rtol = 1e-10, atol = 1e-18)
    @assert isapprox(L_global^2, L_perk^2 + L_disp;    rtol = 1e-10, atol = 1e-18)

    return (; L_abs, L_global, L_perk, L_disp,
              misgauge = c_global^2, c_global, c_k)
end

# ---------------------------------------------------------------------------
# Hamiltonian RMSE (list version)
# ---------------------------------------------------------------------------

function RMSE_H(Glist::AbstractVector,
                λ_ref_list::AbstractVector,
                ψ_ref_list::AbstractVector,
                proj_list::AbstractVector,
                kw_list::AbstractVector,
                filter_list::AbstractVector,
                model, ps, st;
                cone_w = 1.0,
                filter_σ = 0.5,
                cone_gauge::Symbol = :global,
                n_occ = nothing)

    n = length(Glist)
    @assert length(λ_ref_list)  == n
    @assert length(ψ_ref_list)  == n
    @assert length(proj_list)   == n
    @assert length(kw_list)     == n
    @assert length(filter_list) == n

    total = 0.0

    for i in eachindex(Glist)
        total += RMSE_H(Glist[i],
                        λ_ref_list[i],
                        ψ_ref_list[i],
                        proj_list[i],
                        kw_list[i],
                        filter_list[i],
                        model, ps, st;
                        cone_w,
                        filter_σ,
                        cone_gauge,
                        n_occ)
    end

    return sqrt(total / n)
end

# ---------------------------------------------------------------------------
# FermiFilter: a smoothed step function with support near the Fermi level.
# ---------------------------------------------------------------------------

struct FermiFilter{T<:Real}
    μ::T       # Fermi level
    σ::T       # smoothing parameter
    width::T   # half-width of the plateau
    β::T       # inverse temperatures
end

FermiFilter(width::Real) = FermiFilter(0.0, 0.5, width, 10.0)

set_new_filter(σ, f::FermiFilter) = FermiFilter(f.μ, σ, f.width, f.β)

function (f::FermiFilter)(x::Real)
    rise = 0.5 * (1 + erf((x - f.μ + f.width) / f.σ))
    fall = 0.5 * (1 - erf((x - f.μ - f.width) / f.σ))
    return rise + fall - 1
end

# ---------------------------------------------------------------------------
# Hamiltonian RMSE (single structure)
# ---------------------------------------------------------------------------

# `cone_gauge` controls how the high-symmetry / cone eigenvalues are compared:
#   :global   one shift shared by every cone k point and every cone band
#   :absolute no shift
#   :perk     one shift per k point -- diagnostic only, do not train on it
#
# Note the interaction with the Δop term below: that term is gauge sensitive
# through `filter.(real.(λ))`, whose μ is the *absolute* DFT εF.  So with
# `cone_gauge = :global` the absolute energy zero is pinned by the occupied
# subspace term alone, which is the physically meaningful constraint, while
# the band shapes are judged shift-free.
function RMSE_H(G, λ_ref, ψ_ref, proj, kweights, filter, model, ps, st;
                cone_w = 1.0,
                filter_σ = 0.5,
                cone_gauge::Symbol = :global,
                n_occ = nothing)

    result = model(G, ps, st)
    out    = result[1]
    Wb     = out.Wb
    nk     = length(kweights)

    filter = set_new_filter(filter_σ, filter)

    fλ_ref = [filter.(real.(λ)) for λ in λ_ref[1:nk]]
    norm2_ref = [sum(abs2, f) for f in fλ_ref]

    Δop = 0.0

    for (ik, ikw) in enumerate(kweights)
        λ_pred = out.λ[ik]

        Jk = proj[ik] * Wb'

        # CPU path:
        #     out.Sout isa Vector{Matrix}, _kmat(out.Sout, ik) == out.Sout[ik]
        #
        # GPU dense-assembly path:
        #     out.Sout isa CuArray{ComplexF32,3}; out.Sout[ik] is scalar
        #     linear indexing and is invalid with GPUArrays.allowscalar(false).
        Sk  = _kmat(out.Sout, ik)
        iSk = inv(Sk)

        fλ_pred = filter.(real.(λ_pred))

        ψ_pred = Jk * iSk * out.Vlist[ik]

        S_pred = ψ_pred' * ψ_pred
        S_ref_pred = ψ_ref[ik]' * ψ_pred

        # Do not form Diagonal(fλ_pred) and multiply it with CuArray matrices.
        # That path can trigger scalar indexing in LinearAlgebra/ChainRules.
        norm2_pred = _diag_weighted_gram_sq(fλ_pred, S_pred)
        cross_term = _diag_cross_weighted_gram(fλ_ref[ik], fλ_pred, S_ref_pred)

        # Use ikw from enumerate(kweights), not kweights[ik].
        # The latter causes scalar indexing if kweights is ever a CuArray.
        Δop += ikw * (norm2_ref[ik] + norm2_pred - 2cross_term)
    end

    # Train the Dirac cone.
    Δcone = 0.0

    if cone_w > 0
        ktb = div(size(Wb, 1), 2)
        nbands = min(3, ktb)

        # kpw is the index of the highest band at or below μ.  Two things were
        # wrong here.  (1) findlast returns nothing when every band at that k
        # lies above μ, and kpw - nbands + 1 then throws a MethodError instead
        # of saying what happened.  (2) it is read off λ_ref[1], a
        # TRAINING-GRID k point, and then used as the band index at the CONE
        # k points.  The occupied count can differ between them -- at a Dirac
        # point the band exactly at μ flips which side of `x <= 0` it falls on
        # -- and the reference window then sits on the wrong bands.  The
        # model's own window is the fixed ktb ± nbands, so there is no single
        # right answer to substitute silently; pass n_occ explicitly to pin it,
        # otherwise the old behaviour is kept and the hazard is reported.
        kpw = n_occ
        if kpw === nothing
            kpw = findlast(x -> x <= 0, real.(λ_ref[1]) .- filter.μ)
            kpw === nothing && error(
                "RMSE_H: no band at or below μ = $(filter.μ) at k point 1, so " *
                "the cone window cannot be centred.  Pass n_occ = <band index>.")
            for c in (nk+1):length(λ_ref)
                k_c = findlast(x -> x <= 0, real.(λ_ref[c]) .- filter.μ)
                if k_c !== nothing && k_c != kpw
                    @warn "RMSE_H: occupied-band count differs between k point 1 " *
                          "($kpw) and cone k point $(c-nk) ($k_c); the cone " *
                          "reference window is taken at $kpw.  Pass n_occ to pin it." maxlog=1
                    break
                end
            end
        end
        kpw - nbands + 1 >= 1 && kpw + nbands <= length(λ_ref[1]) || error(
            "RMSE_H: cone window $(kpw-nbands+1):$(kpw+nbands) falls outside the " *
            "$(length(λ_ref[1])) reference bands.")

        λ_pred_cone = getindex.(out.λ[nk+1:end], Ref(ktb-nbands+1:ktb+nbands))
        λ_ref_cone = getindex.(λ_ref[nk+1:end], Ref(kpw-nbands+1:kpw+nbands))

        res_cone = λ_pred_cone .- λ_ref_cone
        w_cone   = _uniform_weights(res_cone)
        Δcone    = sum(v -> sum(abs2, v),
                       _apply_gauge(res_cone, w_cone, cone_gauge))

        nsym = length(λ_ref) - nk
        Δcone /= 2nbands * nsym
    end

    # Train the orthogonality for compressed orbitals.
    # orth_w = 0.5
    # Δorth = 0.0
    # if orth_w > 0
    #     Δorth = sum(x -> norm(x - I), out.Sout) / length(out.Hout)
    # end

    return Δop + cone_w * Δcone
end
