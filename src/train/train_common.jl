using LinearAlgebra
using StaticArrays
using ChainRulesCore
using Zygote
using Optimisers: destructure

export FermiFilter, RMSE, RMSE_H

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
# Eigenvalue RMSE
# ---------------------------------------------------------------------------

function RMSE(G, λ_ref, model, ps, st; keep_bands::Int = 4)
    result = model(G, ps, st)
    out = result[1]
    λ_pred = out.λ

    if any(v -> !all(isfinite, v), λ_pred)
        return eltype(first(λ_pred))(1e6)
    end

    λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))
    λ_ref_keep = getindex.(λ_ref, Ref(1:keep_bands))
    sq_diff = sum(v -> sum(abs2, v), λ_pred_keep .- λ_ref_keep)
    n_total = length(λ_pred) * keep_bands
    return sqrt(sq_diff / n_total)
end

function RMSE(Glist::AbstractVector, λ_ref_list::AbstractVector,
              model, ps, st; keep_bands::Int = 4)
    @assert length(Glist) == length(λ_ref_list)

    total_sq = 0.0
    total_n  = 0

    for i in eachindex(Glist)
        result = model(Glist[i], ps, st)
        out    = result[1]
        λ_pred = out.λ

        if any(v -> !all(isfinite, v), λ_pred)
            return eltype(first(λ_pred))(1e6)
        end

        λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))
        λ_ref_keep = getindex.(λ_ref_list[i], Ref(1:keep_bands))
        diff        = λ_pred_keep .- λ_ref_keep
        total_sq   += sum(v -> sum(abs2, v), diff)
        total_n    += length(diff) * keep_bands
    end

    return sqrt(total_sq / total_n)
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
                filter_σ = 0.5)

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
                        filter_σ)
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

function RMSE_H(G, λ_ref, ψ_ref, proj, kweights, filter, model, ps, st;
                cone_w = 1.0,
                filter_σ = 0.5)

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
        kpw = findlast(x -> x <= 0, λ_ref[1] .- filter.μ)
        ktb = div(size(Wb, 1), 2)

        nbands = min(3, ktb)

        λ_pred_cone = getindex.(out.λ[nk+1:end], Ref(ktb-nbands+1:ktb+nbands))
        λ_ref_cone = getindex.(λ_ref[nk+1:end], Ref(kpw-nbands+1:kpw+nbands))

        Δcone = sum(v -> sum(abs2, v), λ_pred_cone .- λ_ref_cone)

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
