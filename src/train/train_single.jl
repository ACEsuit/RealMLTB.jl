using LinearAlgebra: norm
using Zygote
using Optim
using Optimisers: destructure
import LuxCore

export train

function _adam_warmup(lossfun, ps;
    adam_iterations::Int = 200,
    adam_lr::Float64 = 1e-3,
    grad_clip::Float64 = 10.0,
    patience::Int = 100,
    β1::Float64 = 0.9,
    β2::Float64 = 0.999,
    ε::Float64 = 1e-8,
    print_every::Int = 50,
)
    println("    [Phase 1 - Adam warm-up: $adam_iterations steps, lr=$adam_lr]")

    θ0_raw, re = destructure(ps)
    Tparam = eltype(θ0_raw)
    θ = Float64.(θ0_raw)

    m = zeros(Float64, length(θ))
    v = zeros(Float64, length(θ))

    best_loss = Inf
    best_θ = copy(θ)
    history = Float64[]
    no_improve = 0

    f(θvec) = begin
        ps_cur = re(Tparam.(θvec))
        Float64(lossfun(ps_cur))
    end

    for t in 1:adam_iterations
        val, back = Zygote.pullback(f, θ)
        g = first(back(1.0))

        gnorm = norm(g)
        if gnorm > grad_clip && gnorm > 0
            g = g .* (grad_clip / gnorm)
        end

        m .= β1 .* m .+ (1 - β1) .* g
        v .= β2 .* v .+ (1 - β2) .* (g .^ 2)

        mhat = m ./ (1 - β1^t)
        vhat = v ./ (1 - β2^t)

        θ .-= adam_lr .* mhat ./ (sqrt.(vhat) .+ ε)

        push!(history, val)

        if val < best_loss
            best_loss = val
            best_θ .= θ
            no_improve = 0
        else
            no_improve += 1
        end

        if t % print_every == 0 || t == 1
            println("      Adam step $t  loss = $(round(val, digits=6))  |g| = $(round(gnorm, digits=6))")
        end

        if no_improve >= patience
            println("      Adam early stop at step $t")
            break
        end
    end

    best_ps = re(Tparam.(best_θ))
    return best_ps, best_loss, history
end

function _bfgs_refine(lossfun, ps; iterations::Int = 1000)
    println("    [BFGS: $iterations steps]")

    x0_raw, re = destructure(ps)
    Tparam = eltype(x0_raw)
    x0 = Float64.(x0_raw)

    f(x) = Float64(lossfun(re(Tparam.(x))))

    function g!(storage, x)
        g = Zygote.gradient(xx -> lossfun(re(Tparam.(xx))), x)[1]
        storage .= Float64.(g)
    end

    options = Optim.Options(
        g_tol       = 1e-6,
        f_abstol    = 1e-10,
        f_reltol    = 1e-10,
        iterations  = iterations,
        show_trace  = true,
        store_trace = true,
    )

    result  = Optim.optimize(f, g!, x0, Optim.BFGS(), options)
    x_opt   = Optim.minimizer(result)
    ps_cur  = re(Tparam.(x_opt))
    history = [tr.value for tr in result.trace]

    println("      final loss = $(round(f(x_opt), digits=6))")
    return ps_cur, history, result
end

function train(data, model, ps, st, keep_bands, λ_ref;
    iterations::Int        = 1000,
    adam_iterations::Int   = 200,
    adam_lr::Float64       = 1e-3,
    grad_clip::Float64     = 10.0,
    adam_patience::Int     = 100,
    ψ_ref                  = nothing,
    proj                   = nothing,
    kweights               = nothing,
    filter        = nothing,
    H_iterations::Int      = 500,
)
    # ------------------------------------------------------------------------
    # Phase 1 / 2: eigenvalue RMSE
    # ------------------------------------------------------------------------
    loss_rmse(p) = RMSE(data, λ_ref, model, p, st; keep_bands=keep_bands)

    ps_cur, adam_best_loss, hist1 = _adam_warmup(
        loss_rmse, ps;
        adam_iterations = adam_iterations,
        adam_lr         = adam_lr,
        grad_clip       = grad_clip,
        patience        = adam_patience,
    )

    println("    Phase 1 done. Best RMSE = $(round(adam_best_loss, digits=6))")

    ps_cur, hist2, _ = _bfgs_refine(
        loss_rmse, ps_cur;
        iterations = iterations,
    )

    x_history = vcat(hist1, hist2)

    # ------------------------------------------------------------------------
    # Phase 3: PW-space Hamiltonian loss
    # ------------------------------------------------------------------------
    if ψ_ref !== nothing &&
       proj !== nothing &&
       kweights !== nothing &&
       filter !== nothing

        loss_rmse_H(p) = RMSE_H(
            data,
            λ_ref,
            ψ_ref,
            proj,
            kweights,
            filter,
            model, p, st,
        )

        ps_cur, hist3, _ = _bfgs_refine(
            loss_rmse_H, ps_cur;
            iterations = H_iterations,
        )

        x_history = vcat(x_history, hist3)
    else
        println("    [Phase 3 skipped — ψ_ref / proj / kweights / filter not fully provided]")
    end

    return ps_cur, x_history
end