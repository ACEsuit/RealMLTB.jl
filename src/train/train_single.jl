using Zygote
using Optim
using Optimisers
using Optimisers: destructure
import LuxCore

export train

# ===========================================================================
# Single-sample train
#
# Phase 1: Adam warm-up        — eigenvalue loss (λ)
# Phase 2: BFGS refinement     — eigenvalue loss (λ)
# Phase 3: BFGS fine-tune      — Hamiltonian operator-norm loss (RMSE_H)
#          only runs if H_ref_AO and filter are provided
# ===========================================================================
function train(G, model, ps, st, keep_bands, λ_ref;
               iterations::Int      = 1000,
               adam_iterations::Int = 200,
               adam_lr::Float64     = 1e-3,
               grad_clip::Float64   = 10.0,
               λ_reg                = 1e-4,
               # --- Phase 3 kwargs (optional) ---
               H_ref_AO             = nothing,   # reference H in AO basis
               filter::Union{Function,Nothing} = nothing,  # spectral filter
               H_iterations::Int    = 500,        # BFGS steps for Phase 3
               H_reg                = 1e-4)

    # -----------------------------------------------------------------------
    # Phase 1: Adam warm-up (eigenvalue loss)
    # -----------------------------------------------------------------------
    println("    [Phase 1 - Adam warm-up: $adam_iterations steps, lr=$adam_lr]")
    opt       = Optimisers.Adam(adam_lr)
    opt_state = Optimisers.setup(opt, ps)
    ps_cur    = ps

    for step in 1:adam_iterations
        val, gs = Zygote.withgradient(
            p -> RMSE(G, λ_ref, model, p, st; keep_bands=keep_bands, λ_reg=λ_reg),
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

    # -----------------------------------------------------------------------
    # Phase 2: BFGS refinement (eigenvalue loss)
    # -----------------------------------------------------------------------
    println("    [Phase 2 - BFGS refinement: $iterations steps]")
    x0_f32, re = destructure(ps_cur)
    x0 = Float64.(x0_f32)

    f2(x) = Float64(RMSE(G, λ_ref, model, re(eltype(x0_f32).(x)), st;
                         keep_bands=keep_bands, λ_reg=λ_reg))
    function g2!(storage, x)
        g = Zygote.gradient(
            xx -> RMSE(G, λ_ref, model, re(eltype(x0_f32).(xx)), st;
                       keep_bands=keep_bands, λ_reg=λ_reg), x)[1]
        storage .= Float64.(g)
    end

    options2 = Optim.Options(
        g_tol       = 1e-6,
        f_abstol    = 1e-10,
        f_reltol    = 1e-10,
        iterations  = iterations,
        show_trace  = true,
        store_trace = true,
    )
    result2   = Optim.optimize(f2, g2!, x0, Optim.BFGS(), options2)
    x_opt     = Optim.minimizer(result2)
    ps_cur    = re(x_opt)
    x_history = [result2.trace[i].value for i in 1:length(result2.trace)]

    # -----------------------------------------------------------------------
    # Phase 3: BFGS fine-tune (Hamiltonian operator-norm loss)
    # Only runs when H_ref_AO and filter are supplied.
    # -----------------------------------------------------------------------
    if H_ref_AO !== nothing && filter !== nothing
        println("    [Phase 3 - BFGS fine-tune with RMSE_H: $H_iterations steps]")

        x0_H, re_H = destructure(ps_cur)
        x0_H = Float64.(x0_H)

        f3(x) = Float64(RMSE_H(G, H_ref_AO, model,
                                re_H(eltype(x0_H).(x)), st, filter; H_reg=H_reg))
        function g3!(storage, x)
            g = Zygote.gradient(
                xx -> RMSE_H(G, H_ref_AO, model,
                             re_H(eltype(x0_H).(xx)), st, filter; H_reg=H_reg), x)[1]
            storage .= Float64.(g)
        end

        options3 = Optim.Options(
            g_tol       = 1e-6,
            f_abstol    = 1e-10,
            f_reltol    = 1e-10,
            iterations  = H_iterations,
            show_trace  = true,
            store_trace = true,
        )
        result3    = Optim.optimize(f3, g3!, x0_H, Optim.BFGS(), options3)
        x_opt_H    = Optim.minimizer(result3)
        ps_cur     = re_H(x_opt_H)
        x_history  = vcat(x_history,
                          [result3.trace[i].value for i in 1:length(result3.trace)])

        println("    Phase 3 done. Final RMSE_H = $(round(f3(x_opt_H), digits=6))")
    else
        println("    [Phase 3 skipped — H_ref_AO or filter not provided]")
    end

    return ps_cur, x_history
end

# ===========================================================================
# Multi-sample train
#
# Same three-phase structure, just loops over Glist / H_ref_AO_list.
# ===========================================================================
function train(Glist::AbstractVector, model, ps, st, keep_bands, λ_ref_list;
               iterations::Int      = 1000,
               adam_iterations::Int = 200,
               adam_lr::Float64     = 1e-3,
               grad_clip::Float64   = 10.0,
               λ_reg                = 1e-4,
               # --- Phase 3 kwargs (optional) ---
               H_ref_AO_list        = nothing,
               filter::Union{Function,Nothing} = nothing,
               H_iterations::Int    = 500,
               H_reg                = 1e-4)

    # -----------------------------------------------------------------------
    # Phase 1: Adam warm-up (eigenvalue loss)
    # -----------------------------------------------------------------------
    println("    [Phase 1 - Adam warm-up: $adam_iterations steps, lr=$adam_lr]")
    opt       = Optimisers.Adam(adam_lr)
    opt_state = Optimisers.setup(opt, ps)
    ps_cur    = ps

    for step in 1:adam_iterations
        val, gs = Zygote.withgradient(
            p -> RMSE(Glist, λ_ref_list, model, p, st;
                      keep_bands=keep_bands, λ_reg=λ_reg),
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

    # -----------------------------------------------------------------------
    # Phase 2: BFGS refinement (eigenvalue loss)
    # -----------------------------------------------------------------------
    println("    [Phase 2 - BFGS refinement: $iterations steps]")
    x0, re = destructure(ps_cur)

    f2(x) = RMSE(Glist, λ_ref_list, model, re(x), st;
                 keep_bands=keep_bands, λ_reg=λ_reg)
    function g2!(storage, x)
        storage .= Zygote.gradient(f2, x)[1]
    end

    options2 = Optim.Options(
        g_tol       = 1e-6,
        f_abstol    = 1e-10,
        f_reltol    = 1e-10,
        iterations  = iterations,
        show_trace  = true,
        store_trace = true,
    )
    result2   = Optim.optimize(f2, g2!, x0, Optim.BFGS(), options2)
    x_opt     = Optim.minimizer(result2)
    ps_cur    = re(x_opt)
    x_history = [result2.trace[i].value for i in 1:length(result2.trace)]

    # -----------------------------------------------------------------------
    # Phase 3: BFGS fine-tune (Hamiltonian operator-norm loss)
    # -----------------------------------------------------------------------
    if H_ref_AO_list !== nothing && filter !== nothing
        println("    [Phase 3 - BFGS fine-tune with RMSE_H: $H_iterations steps]")

        x0_H, re_H = destructure(ps_cur)
        x0_H = Float64.(x0_H)

        f3(x) = Float64(RMSE_H(Glist, H_ref_AO_list, model,
                                re_H(eltype(x0_H).(x)), st, filter; H_reg=H_reg))
        function g3!(storage, x)
            g = Zygote.gradient(
                xx -> RMSE_H(Glist, H_ref_AO_list, model,
                             re_H(eltype(x0_H).(xx)), st, filter; H_reg=H_reg), x)[1]
            storage .= Float64.(g)
        end

        options3 = Optim.Options(
            g_tol       = 1e-6,
            f_abstol    = 1e-10,
            f_reltol    = 1e-10,
            iterations  = H_iterations,
            show_trace  = true,
            store_trace = true,
        )
        result3   = Optim.optimize(f3, g3!, x0_H, Optim.BFGS(), options3)
        x_opt_H   = Optim.minimizer(result3)
        ps_cur    = re_H(x_opt_H)
        x_history = vcat(x_history,
                         [result3.trace[i].value for i in 1:length(result3.trace)])

        println("    Phase 3 done. Final RMSE_H = $(round(f3(x_opt_H), digits=6))")
    else
        println("    [Phase 3 skipped — H_ref_AO_list or filter not provided]")
    end

    return ps_cur, x_history
end