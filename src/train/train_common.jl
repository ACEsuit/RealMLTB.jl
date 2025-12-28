export RMSE

# --- single-sample RMSE ---
function RMSE(G, λ_ref, model, ps, st; keep_bands = 4)
    λ_pred = model(G, ps, st)[1]
    λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))
    return sqrt(norm(λ_pred_keep .- λ_ref)^2/length(λ_pred))
end

# --- multi-sample RMSE over a list ---
function RMSE(Glist::AbstractVector, λ_ref_list::AbstractVector,
              model, ps, st; keep_bands = 4)
    @assert length(Glist) == length(λ_ref_list)

    sumsq = 0.0
    n     = 0

    for (G, λ_ref) in zip(Glist, λ_ref_list)
        λ_pred = model(G, ps, st)[1]
        λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))

        diff = λ_pred_keep .- λ_ref
        sumsq += norm(diff)^2
        n     += length(diff)
    end

    return sqrt(sumsq / n)
end