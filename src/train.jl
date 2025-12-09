export RMSE, train

function RMSE(G, λ_ref, model, ps, st; keep_bands = 4)
    λ_pred = model(G, ps, st)[1]
    λ_pred_keep_bands = getindex.(λ_pred, Ref(1:keep_bands))
    return norm(λ_pred_keep_bands - λ_ref)
end

function train(G, model, ps, st, keep_bands, λ_ref; iterations = 1000)
    # Flatten parameters
    x0, re = destructure(ps)
    f(x) = RMSE(G, λ_ref, model, re(x), st; keep_bands = keep_bands)
    function g!(storage, x)
        storage .= Zygote.gradient(x -> f(x), x)[1]
    end

    options = Optim.Options(g_tol = 1e-8, f_abstol = 1e-10, f_reltol = 1e-10, iterations = iterations,
                        show_trace = true, store_trace = true)
    result = optimize(f, g!, x0, BFGS(), options)

    trace = result.trace
    x_history = [trace[i].value for i = 1:length(trace)]
    # Extract optimized parameters
    x_opt = Optim.minimizer(result)
    ps = re(x_opt)
    return ps, x_history
end
