# ---------------------------------------------------------------------------
# Band-only training.
#
# This is the objective to use when the thing you want is a band structure and
# the only reference data you have is band energies: no DFT wavefunctions, no
# plane-wave projectors, no Fermi filter, no Dirac-cone term.  Everything the
# subspace loss RMSE_H needs beyond eigenvalues is dropped.
#
# What is kept:
#
#   * the one-global-shift gauge.  A band structure is determined only up to
#     the absolute energy zero of the reference calculation, so ONE scalar is
#     profiled out analytically -- shared by every k point and every band.
#     Per-k shifts are available for diagnosis only and must never be trained
#     on: they discard the k-to-k alignment, which is physical.
#
#   * independent band windows for the model and the reference, so an NB-band
#     model can be fitted to a window of a DFT calculation with more bands.
#
#   * k weights and per-band weights.
#
# What is deliberately NOT required: rotational covariance.  H_out = Wb H_AO Wb'
# with the dense Wb that ACEv3Center initialises is not invariant under a
# general rotation of the structure (measured: 41-56 eV at Gamma on 2-atom Si,
# while a multiplet-complete Wb gives 5e-9 meV).  A band fit in a FIXED frame
# is unaffected -- the fitted model simply belongs to that frame.  It does mean
# you cannot rotate a structure and expect the same bands, and you cannot
# augment the training set with rotated copies.  `train_bands` says so once.
#
# Gauge algebra, used throughout.  With residuals r_kn = λ_pred - λ_ref and
# weights w_k (k) and b_n (band), define per gauge group g
#
#     W_g  = Σ_{k,n in g} w_k b_n
#     S1_g = Σ_{k,n in g} w_k b_n r_kn
#     S2_g = Σ_{k,n in g} w_k b_n r_kn^2
#
# then the optimal shift for that group is c_g = -S1_g / W_g and
#
#     L^2 = Σ_g (S2_g - S1_g^2 / W_g) / Σ_g W_g .
#
# :absolute uses no groups (L^2 = Σ S2 / Σ W), :global one group per structure,
# :global_dataset one group for the whole list, :perk one group per k point.
# This form needs a single pass and no mutation, so it differentiates cleanly.
# ---------------------------------------------------------------------------

export RMSE_bands, band_report, train_bands

const BAND_GAUGES = (:absolute, :global, :global_dataset, :perk)

# S2 - S1^2/W is a variance: non-negative in exact arithmetic, but it is a
# difference of two nearly equal numbers, so it lands at -1e-33 when the fit is
# almost perfect and sqrt() then throws a DomainError.  Clamp at zero.
@inline _nnsqrt(x) = sqrt(max(x, zero(x)))

"""
    _band_sums(res, w, b) -> (S1k, S2k, Wk)

Per-k-point moments, each a plain vector so that nothing here broadcasts over
tuples -- Zygote's tuple-broadcast pullback returns a Thunk that
ZygoteRules.gradtuple1 rejects ("Gradient Thunk(...) should be a tuple"), so
the moments must stay in separate arrays.
"""
function _band_sums(res, w, b)
    S1k = [w[ik] * sum(b .* res[ik])          for ik in eachindex(res)]
    S2k = [w[ik] * sum(b .* (res[ik] .^ 2))   for ik in eachindex(res)]
    Wk  = [w[ik] * sum(b)                     for ik in eachindex(res)]
    return S1k, S2k, Wk
end

"""
    _band_groups(res, w, b, gauge) -> (Σ_g (S2_g - S1_g^2/W_g), Σ_g W_g)

`res` is a vector over k of residual vectors, `w` the k weights, `b` the band
weights.  Returns the gauge-reduced squared sum and the total weight.
"""
function _band_groups(res, w, b, gauge::Symbol)
    S1k, S2k, Wk = _band_sums(res, w, b)
    Wt = sum(Wk)

    gauge === :absolute && return sum(S2k), Wt
    gauge === :perk     && return sum(S2k .- (S1k .^ 2) ./ Wk), Wt
    # :global and :global_dataset are both "one group over what was handed in";
    # the list method decides which by how it calls this.
    (gauge === :global || gauge === :global_dataset) &&
        return sum(S2k) - sum(S1k)^2 / Wt, Wt

    error("unknown gauge $gauge; expected one of $BAND_GAUGES")
end

@inline function _band_window(λ, bands)
    bands === nothing && return λ
    return λ[bands]
end

function _band_residuals(λ_pred, λ_ref, bands, ref_bands)
    rb = ref_bands === nothing ? bands : ref_bands
    nk = length(λ_pred)
    @assert nk <= length(λ_ref) "model has $nk k points, reference has $(length(λ_ref))"
    return [real.(_band_window(λ_pred[ik], bands)) .-
            real.(_band_window(λ_ref[ik], rb)) for ik in 1:nk]
end

function _band_weight_vector(res, band_weights)
    n = length(first(res))
    band_weights === nothing && return ones(Float64, n)
    @assert length(band_weights) == n "band_weights has $(length(band_weights)) entries, the window has $n bands"
    return Float64.(band_weights)      # not collect(): that calls copyto!,
                                       # which Zygote refuses to trace
end

_band_kweights(res, kweights) =
    kweights === nothing ? ones(Float64, length(res)) : Float64.(kweights)

# ---------------------------------------------------------------------------
# Single structure
# ---------------------------------------------------------------------------

"""
    RMSE_bands(G, λ_ref, model, ps, st; kwargs...)

Band-energy RMSE of one structure, in the units of `λ_ref`.

  bands         UnitRange (or any index vector) into the MODEL's eigenvalues.
                `nothing` uses all of them.
  ref_bands     the matching window in the REFERENCE.  Defaults to `bands`.
                Give it explicitly when the model's band 1 is not the
                reference's band 1.
  kweights      weight per k point; `nothing` is uniform.
  band_weights  weight per band inside the window; `nothing` is uniform.
  gauge         :global (default), :absolute or :perk.  See the header.
"""
function RMSE_bands(G, λ_ref, model, ps, st;
                    bands = nothing,
                    ref_bands = nothing,
                    kweights = nothing,
                    band_weights = nothing,
                    gauge::Symbol = :global,
                    burg_w::Real = 0.0)

    out = model(G, ps, st)[1]
    λ_pred = out.λ

    if any(v -> !all(isfinite, v), λ_pred)
        return eltype(real(first(first(λ_pred))))(1e6)
    end

    res = _band_residuals(λ_pred, λ_ref, bands, ref_bands)
    w   = _band_kweights(res, kweights)
    b   = _band_weight_vector(res, band_weights)

    sq, Wt = _band_groups(res, w, b, gauge)

    # The barrier is added to the SQUARED objective, so the returned number
    # stays in the units of λ_ref and reduces exactly to the band RMSE at
    # burg_w = 0.  See src/train/burg.jl for what it is and why the weight has
    # to be calibrated by the gradient ratio rather than guessed.
    pen = burg_w > 0 ? burg_penalty(out.Sout; weight = burg_w) : zero(sq / Wt)
    return _nnsqrt(sq / Wt + pen)
end

# ---------------------------------------------------------------------------
# Many structures
# ---------------------------------------------------------------------------

"""
    RMSE_bands(Glist, λ_ref_list, model, ps, st; shift_scope = :structure, ...)

Band-energy RMSE over a list of structures.

`shift_scope` decides what the one global shift is shared over:

  :structure  one shift per structure (default).  Use this when each reference
              calculation carries its own arbitrary energy zero.
  :dataset    a single shift for the whole list.  Stricter, and correct when
              every reference came from the same code with the same
              pseudopotentials and the same G = 0 convention.

Ignored for `gauge = :absolute` and `gauge = :perk`.
"""
function RMSE_bands(Glist::AbstractVector, λ_ref_list::AbstractVector,
                    model, ps, st;
                    bands = nothing,
                    ref_bands = nothing,
                    kweights_list = nothing,
                    band_weights = nothing,
                    gauge::Symbol = :global,
                    shift_scope::Symbol = :structure,
                    burg_w::Real = 0.0)

    @assert length(Glist) == length(λ_ref_list)
    @assert shift_scope in (:structure, :dataset) "shift_scope must be :structure or :dataset"

    per = map(eachindex(Glist)) do i
        out = model(Glist[i], ps, st)[1]
        λ_pred = out.λ

        if any(v -> !all(isfinite, v), λ_pred)
            return [NaN, NaN, NaN, NaN]
        end

        pen_i = burg_w > 0 ? burg_penalty(out.Sout; weight = burg_w) : 0.0

        res = _band_residuals(λ_pred, λ_ref_list[i], bands, ref_bands)
        w   = kweights_list === nothing ? ones(Float64, length(res)) :
                                          Float64.(kweights_list[i])
        b   = _band_weight_vector(res, band_weights)

        S1k, S2k, Wk = _band_sums(res, w, b)

        # An ARRAY, not a 4-tuple: for a tuple returned from a differentiated
        # map Zygote builds a Tangent{NTuple{4,Float64}} that it cannot add
        # back into the enclosing tangent, which makes this method's gradient
        # fail where the single-structure one above succeeds.
        if gauge === :perk
            return [0.0, sum(S2k .- (S1k .^ 2) ./ Wk), sum(Wk), pen_i]
        end
        return [sum(S1k), sum(S2k), sum(Wk), pen_i]
    end

    S1v = [p[1] for p in per]
    S2v = [p[2] for p in per]
    Wv  = [p[3] for p in per]
    pen = sum(p[4] for p in per) / length(per)

    any(isnan, Wv) && return 1e6
    Wt = sum(Wv)

    if gauge === :absolute || gauge === :perk
        return _nnsqrt(sum(S2v) / Wt + pen)
    elseif gauge === :global && shift_scope === :structure
        return _nnsqrt(sum(S2v .- (S1v .^ 2) ./ Wv) / Wt + pen)
    elseif gauge === :global || gauge === :global_dataset
        return _nnsqrt((sum(S2v) - sum(S1v)^2 / Wt) / Wt + pen)
    end
    error("unknown gauge $gauge; expected one of $BAND_GAUGES")
end

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

"""
    band_report(λ_pred, λ_ref; kwargs...) -> NamedTuple

Everything you want to see about a band fit on one structure, without going
through the model again.  Energies stay in the units of the inputs.

  L_abs, L_global, L_perk    the three gauges
  L_disp                     L_global^2 - L_perk^2, the gauge dispersion
  c_global, c_k              the fitted shifts
  per_band, per_k            gauge-reduced RMSE resolved along each axis
  max_abs_error              worst single (k, band) residual after the global
                             shift, which is the number that actually bites

Identities asserted here: L_abs^2 == L_global^2 + c_global^2 and
L_global^2 == L_perk^2 + L_disp.
"""
function band_report(λ_pred, λ_ref;
                     bands = nothing,
                     ref_bands = nothing,
                     kweights = nothing,
                     band_weights = nothing)

    res = _band_residuals(λ_pred, λ_ref, bands, ref_bands)
    w   = _band_kweights(res, kweights)
    b   = _band_weight_vector(res, band_weights)

    S1k, S2k, Wk = _band_sums(res, w, b)
    S1, S2, Wt = sum(S1k), sum(S2k), sum(Wk)

    c_global = -S1 / Wt
    c_k      = -S1k ./ Wk

    L_abs    = _nnsqrt(S2 / Wt)
    L_global = _nnsqrt((S2 - S1^2 / Wt) / Wt)
    L_perk   = _nnsqrt(sum(S2k .- (S1k .^ 2) ./ Wk) / Wt)
    L_disp   = max(sum(Wk .* ((c_k .- c_global) .^ 2)) / Wt, 0.0)

    scale = max(L_abs^2, eps())
    @assert isapprox(L_abs^2, L_global^2 + c_global^2; rtol = 1e-9, atol = 1e-9scale)
    @assert isapprox(L_global^2, L_perk^2 + L_disp;    rtol = 1e-9, atol = 1e-9scale)

    shifted  = [r .+ c_global for r in res]
    nb       = length(first(res))
    per_band = [_nnsqrt(sum(ik -> w[ik] * shifted[ik][n]^2, eachindex(res)) /
                        sum(w)) for n in 1:nb]
    per_k    = [_nnsqrt(sum(b .* (shifted[ik] .^ 2)) / sum(b)) for ik in eachindex(res)]

    return (; L_abs, L_global, L_perk, L_disp, c_global, c_k,
              per_band, per_k,
              max_abs_error = maximum(v -> maximum(abs, v), shifted))
end

# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

"""
    train_bands(Glist, λ_ref_list, model, ps, st; kwargs...) -> ps, history

Adam warm-up followed by BFGS on `RMSE_bands` alone.  Band energies are the
only reference data needed.

Keywords beyond those of `RMSE_bands`: `adam_iterations`, `adam_lr`,
`grad_clip`, `adam_patience`, `iterations`, `train_part`, `verbose`.
"""
function train_bands(Glist, λ_ref_list, model, ps, st;
                     bands = nothing,
                     ref_bands = nothing,
                     kweights_list = nothing,
                     band_weights = nothing,
                     gauge::Symbol = :global,
                     shift_scope::Symbol = :structure,
                     burg_w::Real = 0.0,
                     adam_iterations::Int = 200,
                     adam_lr::Float64 = 1e-3,
                     grad_clip::Float64 = 10.0,
                     adam_patience::Int = 100,
                     iterations::Int = 1000,
                     train_part::Symbol = :all,
                     precond = nothing,
                     initial_invH = nothing,
                     restart_steps::Int = 10_000,
                     check_overlap::Symbol = :error,
                     verbose::Bool = true)

    # Before anything else: an indefinite S(k) makes the objective Inf at every
    # evaluation, so the fit cannot start.  See check_overlap_pd.
    check_overlap_pd(model, Glist, ps, st; mode = check_overlap,
                     verbose = verbose, label = "  [train_bands] ")

    if verbose
        println("  [train_bands] eigenvalues only: no ψ_ref, no projectors, " *
                "no Fermi filter, no cone term.")
        println("  [train_bands] gauge = $gauge" *
                (gauge === :global ? ", shift_scope = $shift_scope" : "") *
                ".  Bands are fitted up to " *
                (gauge === :absolute ? "no shift at all" :
                 gauge === :perk ? "ONE SHIFT PER K POINT -- diagnostic only" :
                 shift_scope === :dataset ? "one shift for the whole dataset" :
                                            "one shift per structure") * ".")
        burg_w > 0 && println("  [train_bands] Burg barrier on S_out, weight " *
                              "$burg_w.  Calibrate it with burg_report: the " *
                              "gradient ratio eta, not the raw value, is what matters.")
        println("  [train_bands] rotational covariance is NOT imposed: with a " *
                "dense Wb the fit belongs to the frame of the training data. " *
                "Do not augment with rotated copies.")
    end

    lossfun(p) = RMSE_bands(Glist, λ_ref_list, model, p, st;
                            bands, ref_bands, kweights_list, band_weights,
                            gauge, shift_scope, burg_w)

    ps_cur, best, hist1 = _adam_warmup(lossfun, ps;
        train_part = train_part,
        adam_iterations = adam_iterations,
        adam_lr = adam_lr,
        grad_clip = grad_clip,
        patience = adam_patience)

    verbose && println("    Adam done. Best band RMSE = $(round(best, digits=8))")

    ps_cur, hist2, _, invH = _bfgs_refine(lossfun, ps_cur;
        iterations = iterations, train_part = train_part,
        precond = precond, initial_invH = initial_invH,
        restart_steps = restart_steps, show_trace = verbose)

    # invH comes back so a caller running several stages can hand the curvature
    # on instead of cold-starting BFGS again.
    return ps_cur, vcat(hist1, hist2), invH
end
