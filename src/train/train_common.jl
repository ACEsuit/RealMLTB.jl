using Base.Threads
using LinearAlgebra
using StaticArrays
using ChainRulesCore
using Zygote
using Optimisers

export RMSE

# --- single-sample RMSE ---
function RMSE(G, λ_ref, model, ps, st; keep_bands = 4)
    λ_pred = model(G, ps, st)[1]
    λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))
    
    sq_diff = sum(v -> sum(abs2, v), λ_pred_keep .- λ_ref)
    
    N = length(λ_pred) * keep_bands
    return sqrt(sq_diff / N)
end

# --- multi-sample RMSE over a list ---
function RMSE(Glist::AbstractVector, λ_ref_list::AbstractVector,
              model, ps, st; keep_bands = 4)
    @assert length(Glist) == length(λ_ref_list)
    
    num_samples = length(Glist)
    partial_sumsq = zeros(Float64, nthreads())
    partial_counts = zeros(Int, nthreads())

    Threads.@threads for i in 1:num_samples
        tid = threadid()
        G, λ_ref = Glist[i], λ_ref_list[i]
        
        λ_pred = model(G, ps, st)[1]
        λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))
        
        diff = λ_pred_keep .- λ_ref
        
        partial_sumsq[tid] += sum(v -> sum(abs2, v), diff)
        partial_counts[tid] += length(diff) * keep_bands
    end

    return sqrt(sum(partial_sumsq) / sum(partial_counts))
end

# --- ChainRulesCore rrule for Zygote ---
function ChainRulesCore.rrule(::typeof(RMSE), Glist, λ_ref_list, model, ps, st; keep_bands = 4)
    val = RMSE(Glist, λ_ref_list, model, ps, st; keep_bands = keep_bands)
    
    p, re = destructure(ps)

    function RMSE_pullback(Δval)
        N = length(Glist) * length(λ_ref_list[1]) * keep_bands
        
        # dRMSE/dy = (y_pred - y_ref) / (N * RMSE)
        scale = Δval / (N * val)
        
        current_adj_p = zero(p) 

        Threads.@threads for i in 1:length(Glist)
            y_pred_full, pb_model = Zygote.pullback(v -> model(Glist[i], re(v), st)[1], p)
            
            y_grad = [zeros(eltype(p), length(y)) for y in y_pred_full]
            
            for k in 1:length(y_grad)
                diff = y_pred_full[k][1:keep_bands] .- λ_ref_list[i][k]
                y_grad[k][1:keep_bands] .= diff .* scale
            end
            
            grads = pb_model(y_grad)
            d_p_i = grads[1]
            
            if d_p_i !== nothing
                current_adj_p .+= d_p_i
            end
        end

        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), re(current_adj_p), NoTangent())
    end

    return val, RMSE_pullback
end