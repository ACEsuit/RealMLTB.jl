using Zygote
using Optim
import LuxCore

export train

"""
    train(G, model, ps, st, keep_bands, λ_ref; iterations=1000)

Train on a single graph/sample using BFGS.
Returns (ps_opt, x_history).
"""
function train(G, model, ps, st, keep_bands, λ_ref; iterations::Int=1000)
    x0, re = destructure(ps)

    f(x) = RMSE(G, λ_ref, model, re(x), st; keep_bands=keep_bands)

    function g!(storage, x)
        storage .= Zygote.gradient(xx -> f(xx), x)[1]
    end

    options = Optim.Options(
        g_tol = 1e-8,
        f_abstol = 1e-10,
        f_reltol = 1e-10,
        iterations = iterations,
        show_trace = true,
        store_trace = true
    )

    result = Optim.optimize(f, g!, x0, Optim.BFGS(), options)
    trace = result.trace
    x_history = [trace[i].value for i in 1:length(trace)]

    x_opt = Optim.minimizer(result)
    ps_opt = re(x_opt)
    return ps_opt, x_history
end

"""
    train(Glist, model, ps, st, keep_bands, λ_ref_list; iterations=1000)

Train on a list by minimizing RMSE over the list.
Note: This uses threads inside RMSE for speed.
"""
function train(Glist::AbstractVector, model, ps, st, keep_bands, λ_ref_list; iterations::Int=1000)
    x0, re = destructure(ps)

    f(x) = RMSE(Glist, λ_ref_list, model, re(x), st; keep_bands=keep_bands)

    function g!(storage, x)
        storage .= Zygote.gradient(xx -> f(xx), x)[1]
    end

    options = Optim.Options(
        g_tol = 1e-8,
        f_abstol = 1e-10,
        f_reltol = 1e-10,
        iterations = iterations,
        show_trace = true,
        store_trace = true
    )

    result = Optim.optimize(f, g!, x0, Optim.BFGS(), options)
    trace = result.trace
    x_history = [trace[i].value for i in 1:length(trace)]

    x_opt = Optim.minimizer(result)
    ps_opt = re(x_opt)
    return ps_opt, x_history
end
