export RMSE, train

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

# --- train for single sample ---
function train(G, model, ps, st, keep_bands, λ_ref; iterations = 1000)
    x0, re = destructure(ps)
    f(x) = RMSE(G, λ_ref, model, re(x), st; keep_bands = keep_bands)

    function g!(storage, x)
        storage .= Zygote.gradient(xx -> f(xx), x)[1]
    end

    options = Optim.Options(
        g_tol = 1e-8, f_abstol = 1e-10, f_reltol = 1e-10,
        iterations = iterations, show_trace = true, store_trace = true
    )
    result = optimize(f, g!, x0, BFGS(), options)

    trace = result.trace
    x_history = [trace[i].value for i in 1:length(trace)]

    x_opt = Optim.minimizer(result)
    ps = re(x_opt)
    return ps, x_history
end
