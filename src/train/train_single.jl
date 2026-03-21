using Zygote
using Optim
using Optimisers
using Optimisers: destructure
import LuxCore

export train

function train(G, model, ps, st, keep_bands, λ_ref;
               iterations::Int     = 1000,
               adam_iterations::Int= 200,    
               adam_lr::Float64    = 1e-3,   
               grad_clip::Float64  = 10.0, 
               λ_reg = 1e-4  
               )

    # ---- Phase 1: Adam warm-up ----
    println("    [Adam warm-up: $adam_iterations steps, lr=$adam_lr]")
    opt = Optimisers.Adam(adam_lr)
    opt_state = Optimisers.setup(opt, ps)

    ps_cur = ps
    for step in 1:adam_iterations
        val, gs = Zygote.withgradient(
            p -> RMSE(G, λ_ref, model, p, st; keep_bands = keep_bands, λ_reg=λ_reg),
            ps_cur)
        g = gs[1]

        # gradient clipping
        x_g, re_g = destructure(g)
        gnorm = norm(x_g)
        if gnorm > grad_clip
            x_g .*= (grad_clip / gnorm)
            g = re_g(x_g)
        end

        opt_state, ps_cur = Optimisers.update!(opt_state, ps_cur, g)

        if step % 50 == 0 || step == 1
            println("      Adam step $step  loss = $(round(val, digits=4))")
        end
    end

    # ---- Phase 2: BFGS refinement ----
    println("    [BFGS refinement: $iterations steps]")
    x0_f32, re = destructure(ps_cur)
    x0 = Float64.(x0_f32) 

    f(x) = Float64(RMSE(G, λ_ref, model, re(eltype(x0_f32).(x)), st; keep_bands = keep_bands, λ_reg=λ_reg))

    function g!(storage, x)
        g = Zygote.gradient(
            xx -> RMSE(G, λ_ref, model, re(eltype(x0_f32).(xx)), st; keep_bands = keep_bands, λ_reg=λ_reg),
            x)[1]
        storage .= Float64.(g)
    end

    options = Optim.Options(
        g_tol        = 1e-6,
        f_abstol     = 1e-10,
        f_reltol     = 1e-10,
        iterations   = iterations,
        show_trace   = true,
        store_trace  = true,
    )

    result    = Optim.optimize(f, g!, x0, Optim.BFGS(), options)
    trace     = result.trace
    x_history = [trace[i].value for i in 1:length(trace)]
    x_opt     = Optim.minimizer(result)
    ps_opt    = re(x_opt)

    return ps_opt, x_history
end

function train(Glist::AbstractVector, model, ps, st, keep_bands, λ_ref_list;
               iterations::Int    = 1000,
               adam_iterations::Int    = 200,
               adam_lr::Float64   = 1e-3,
               grad_clip::Float64 = 10.0,
               λ_reg = 1e-4  
               )

    # ---- Phase 1: Adam warm-up ----
    println("    [Adam warm-up: $adam_iterations steps, lr=$adam_lr]")
    opt = Optimisers.Adam(adam_lr)
    opt_state = Optimisers.setup(opt, ps)

    ps_cur = ps
    for step in 1:adam_iterations
        val, gs = Zygote.withgradient(
            p -> RMSE(Glist, λ_ref_list, model, p, st; keep_bands = keep_bands, λ_reg = λ_reg),
            ps_cur)
        g = gs[1]

        x_g, re_g = destructure(g)
        gnorm = norm(x_g)
        if gnorm > grad_clip
            x_g .*= (grad_clip / gnorm)
            g = re_g(x_g)
        end

        opt_state, ps_cur = Optimisers.update!(opt_state, ps_cur, g)

        if step % 50 == 0 || step == 1
            println("      Adam step $step  loss = $(round(val, digits=4))")
        end
    end

    # ---- Phase 2: BFGS refinement ----
    println("    [BFGS refinement: $iterations steps]")
    x0, re = destructure(ps_cur)

    f(x) = RMSE(Glist, λ_ref_list, model, re(x), st; keep_bands = keep_bands, λ_reg = λ_reg)
    function g!(storage, x)
        storage .= Zygote.gradient(f, x)[1]
    end

    options = Optim.Options(
        g_tol        = 1e-6,
        f_abstol     = 1e-10,
        f_reltol     = 1e-10,
        iterations   = iterations,
        show_trace   = true,
        store_trace  = true,
    )

    result    = Optim.optimize(f, g!, x0, Optim.BFGS(), options)
    trace     = result.trace
    x_history = [trace[i].value for i in 1:length(trace)]
    x_opt     = Optim.minimizer(result)
    ps_opt    = re(x_opt)

    return ps_opt, x_history
end