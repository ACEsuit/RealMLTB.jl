using LinearAlgebra
using StaticArrays
using ChainRulesCore
using Zygote
using Optimisers: destructure
using MatrixFuns

export RMSE, RMSE_H

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
    sq_diff = sum(v -> sum(abs2, v), λ_pred_keep .- λ_ref)
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
        diff        = λ_pred_keep .- λ_ref_list[i]
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
                model, ps, st)

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
                        model, ps, st)
    end
    return total / n
end

# ---------------------------------------------------------------------------
# FermiFilter
# ---------------------------------------------------------------------------

struct FermiFilter{T<:Real}
    μ::T       # Fermi level
    σ::T       # smoothing parameter
    width::T   # half-width of the plateau
end
FermiFilter() = FermiFilter(0.0, 0.1, 0.2)

function eval_filter(x, f)
    rise = 0.5 * (1 + erf((x - f.μ + f.width) / f.σ))
    fall = 0.5 * (1 - erf((x - f.μ - f.width) / f.σ))
    return rise + fall - 1
end

# ---------------------------------------------------------------------------
# Hamiltonian RMSE (single structure)
# ---------------------------------------------------------------------------

function RMSE_H(G, λ_ref, ψ_ref, proj, kweights, filter, model, ps, st)

    result = model(G, ps, st)
    out    = result[1]
    Wb     = out.Wb

    filter_fn = let f = filter
        x -> eval_filter(x, f)
    end

    Δop = sum(enumerate(kweights)) do (ik, ikw)
        ikw = kweights[ik]

        Jk  = proj[ik] * Wb'
        ihS = ortho_lowdin_hS(Jk)
        Jk  = Jk * ihS
        λ_pred = out.λ[ik]
        V      = out.Vlist[ik]
        V_orth = ihS * V
        fH_k_proj = Jk * (V_orth * Diagonal(filter_fn.(real.(λ_pred))) * V_orth') * Jk'
        fH_k_ref  = mat_fun_eigen(filter_fn, λ_ref[ik], ψ_ref[ik])

        ΔfH_k = fH_k_ref - fH_k_proj
        ikw * norm(ΔfH_k)
    end

    return Δop
end

function mat_fun_eigen(f, λ, ψ)
    fλ = f.(real.(λ))
    return ψ * Diagonal(fλ) * ψ'
end

# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------

function ortho_lowdin_hS(φk::AbstractArray{T}) where {T}
    evals, evecs = eigen(Hermitian(φk' * φk))
    @assert minimum(abs, evals) > eps(real(T)) * maximum(abs, evals)
    return evecs * Diagonal(evals .^ (-0.5)) * evecs'
end
