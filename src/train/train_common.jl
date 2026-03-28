using LinearAlgebra
using StaticArrays
using ChainRulesCore
using Zygote
using Optimisers: destructure

export RMSE

# --- single-sample RMSE ---
function RMSE(G, λ_ref, model, ps, st; keep_bands = 4, λ_reg = 1e-4)
    λ_pred = model(G, ps, st)[1]

    if any(v -> !all(isfinite, v), λ_pred)
        return eltype(first(λ_pred))(1e6)
    end

    λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))
    sq_diff = sum(v -> sum(abs2, v), λ_pred_keep .- λ_ref)
    N       = length(λ_pred) * keep_bands
    rmse    = sqrt(sq_diff / N)

    x, _  = destructure(ps)
    reg   = λ_reg * sum(abs2, x)

    return rmse + reg
end

# --- multi-sample RMSE over a list ---
function RMSE(Glist::AbstractVector, λ_ref_list::AbstractVector,
              model, ps, st; keep_bands = 4, λ_reg = 1e-4)
    @assert length(Glist) == length(λ_ref_list)

    total_sq = 0.0
    total_n  = 0

    for i in eachindex(Glist)
        λ_pred = model(Glist[i], ps, st)[1]

        if any(v -> !all(isfinite, v), λ_pred)
            return eltype(first(λ_pred))(1e6)
        end

        λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))
        diff     = λ_pred_keep .- λ_ref_list[i]
        total_sq += sum(v -> sum(abs2, v), diff)
        total_n  += length(diff) * keep_bands
    end

    rmse  = sqrt(total_sq / total_n)
    x, _  = destructure(ps)
    reg   = λ_reg * sum(abs2, x)

    return rmse + reg
end

using MatrixFuns
# compute $\|f(H(R)) - f(H^{ref}(R))\|_{op}$ with the unit cell $R=0$
# H^{ref}(R)_{Ia, Jb} = \sum_{a',b'} (W^{ψ}_{Ia,a'})' * (H^{proj}(R))_{Ia',Jb'} * W^{ψ}_{Jb,b'}
# H^{proj}(R) = 1/{N_k}\sum_k exp^{-im*k⋅R} * C(k)' * H^{PW}(k) * C(k) 
# with (C(k))_{G, Ia} = \<e^{k+G}|ϕ_{Ia}\>
function RMSE(H, H_ref, Wb, ps, st, filter::Function; H_reg=1e-4)
    H_ref = Wb * H_ref * Wb' 
    fH_ref = P * Diagonal(f.(D)) * inv(P)

    ΔfH = mat_fun(filter, H_ref) - mat_fun(filter, H)
    rmse = opnorm(ΔfH)
end
