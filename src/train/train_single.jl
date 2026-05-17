using LinearAlgebra: norm
using Zygote
using Optim
using Optimisers: destructure
import LuxCore

export train


@inline function _replace_layer1(ps, l1_new)
    return merge(ps, (; layer_1 = l1_new))
end

function _select_trainable_part(ps, train_part::Symbol)
    train_part === :all && return ps, identity

    hasproperty(ps, :layer_1) ||
        error("Expected ps.layer_1. Got parameter tree with properties $(propertynames(ps)).")

    l1 = ps.layer_1

    if train_part in (:potential, :V, :Wv)
        if hasproperty(l1, :ace)
            part = l1.ace
            rebuild = part_new -> _replace_layer1(ps, merge(l1, (; ace = part_new)))
            return part, rebuild

        elseif hasproperty(l1, :acev)
            part = l1.acev
            rebuild = part_new -> _replace_layer1(ps, merge(l1, (; acev = part_new)))
            return part, rebuild

        elseif hasproperty(l1, :Wv)
            part = l1.Wv
            rebuild = part_new -> _replace_layer1(ps, merge(l1, (; Wv = part_new)))
            return part, rebuild

        else
            error("train_part = $train_part requested, but no potential block found in ps.layer_1. Properties: $(propertynames(l1))")
        end
    end

    if train_part in (:basis, :minimal_basis, :Wb, :wpsi)
        if hasproperty(l1, :Wb)
            part = l1.Wb
            rebuild = part_new -> _replace_layer1(ps, merge(l1, (; Wb = part_new)))
            return part, rebuild

        elseif hasproperty(l1, :wpsi)
            part = l1.wpsi
            rebuild = part_new -> _replace_layer1(ps, merge(l1, (; wpsi = part_new)))
            return part, rebuild

        else
            error("train_part = $train_part requested, but no basis block found in ps.layer_1. Properties: $(propertynames(l1))")
        end
    end

    error("Unknown train_part = $train_part. Use :all, :potential, or :basis.")
end

function _make_objective_over_part(lossfun, ps, train_part::Symbol)
    part0, rebuild = _select_trainable_part(ps, train_part)

    θ0_raw, re = destructure(part0)
    length(θ0_raw) > 0 ||
        error("Selected parameter block train_part=$train_part has zero trainable parameters.")

    Tparam = eltype(θ0_raw)
    θ0 = Float64.(θ0_raw)

    function f(θvec)
        part_cur = re(Tparam.(θvec))
        ps_cur = rebuild(part_cur)
        return Float64(lossfun(ps_cur))
    end

    function rebuild_ps(θvec)
        part_cur = re(Tparam.(θvec))
        return rebuild(part_cur)
    end

    return θ0, f, rebuild_ps
end

# ============================================================================
# Adam warmup
# ============================================================================

function _adam_warmup(lossfun, ps;
    train_part::Symbol = :all,
    adam_iterations::Int = 200,
    adam_lr::Float64 = 1e-3,
    grad_clip::Float64 = 10.0,
    patience::Int = 100,
    β1::Float64 = 0.9,
    β2::Float64 = 0.999,
    ε::Float64 = 1e-8,
    print_every::Int = 50,
)
    println("    [Phase 1 - Adam warm-up: $adam_iterations steps, lr=$adam_lr, train_part=$train_part]")

    θ, f, rebuild_ps = _make_objective_over_part(lossfun, ps, train_part)

    m = zeros(Float64, length(θ))
    v = zeros(Float64, length(θ))

    best_loss = Inf
    best_θ = copy(θ)
    history = Float64[]
    no_improve = 0

    for t in 1:adam_iterations
        val, back = Zygote.pullback(f, θ)
        g_raw = first(back(1.0))
        g = g_raw === nothing ? zeros(Float64, length(θ)) : Float64.(g_raw)

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

    best_ps = rebuild_ps(best_θ)
    return best_ps, best_loss, history
end

# ============================================================================
# BFGS refinement
# ============================================================================

function _bfgs_refine(lossfun, ps;
    iterations::Int = 1000,
    train_part::Symbol = :all,
)
    println("    [BFGS: $iterations steps, train_part=$train_part]")

    x0, f, rebuild_ps = _make_objective_over_part(lossfun, ps, train_part)

    function g!(storage, x)
        g_raw = Zygote.gradient(f, x)[1]
        if g_raw === nothing
            fill!(storage, 0.0)
        else
            storage .= Float64.(g_raw)
        end
        return nothing
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
    ps_cur  = rebuild_ps(x_opt)
    history = [tr.value for tr in result.trace]

    println("      final loss = $(round(f(x_opt), digits=6))")
    return ps_cur, history, result
end

# ============================================================================
# Main train routine
# ============================================================================

function train(data, model, ps, st, keep_bands, λ_ref;
    iterations::Int        = 1000,
    adam_iterations::Int   = 200,
    adam_lr::Float64       = 1e-3,
    grad_clip::Float64     = 10.0,
    adam_patience::Int     = 100,

    # New:
    #   :all        train everything
    #   :potential  train V / ACE potential only
    #   :basis      train Wb / Wpsi only
    train_part::Symbol     = :all,
    H_train_part::Symbol   = train_part,

    ψ_ref                  = nothing,
    proj                   = nothing,
    kweights               = nothing,
    filter                 = nothing,
    H_iterations::Int      = 500,
)
    # ------------------------------------------------------------------------
    # Phase 1 / 2: eigenvalue RMSE
    # ------------------------------------------------------------------------
    loss_rmse(p) = RMSE(data, λ_ref, model, p, st; keep_bands = keep_bands)

    ps_cur, adam_best_loss, hist1 = _adam_warmup(
        loss_rmse, ps;
        train_part      = train_part,
        adam_iterations = adam_iterations,
        adam_lr         = adam_lr,
        grad_clip       = grad_clip,
        patience        = adam_patience,
    )

    println("    Phase 1 done. Best RMSE = $(round(adam_best_loss, digits=6))")

    ps_cur, hist2, _ = _bfgs_refine(
        loss_rmse, ps_cur;
        iterations = iterations,
        train_part = train_part,
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
            train_part = H_train_part,
        )

        x_history = vcat(x_history, hist3)
    else
        println("    [Phase 3 skipped — ψ_ref / proj / kweights / filter not fully provided]")
    end

    return ps_cur, x_history
end