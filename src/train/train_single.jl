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

    # S_out = Wb S_AO Wb' loses positive definiteness whenever Wb approaches
    # rank deficiency, and generate_λ_and_vecs then throws from cholesky(S)
    # BEFORE any eigenvalue exists -- so the losses' own isfinite guard cannot
    # see it and the whole run dies on one bad trial step.  Reject the step
    # instead, as the reference optimizer does
    # (`e isa PosDefException || e isa LAPACKException || rethrow(); Inf`).
    # It does NOT replace a rank barrier: a fit that keeps hitting this is
    # being pushed towards a singular Wb and needs the penalty, not the catch.
    function f(θvec)
        part_cur = re(Tparam.(θvec))
        ps_cur = rebuild(part_cur)
        try
            return Float64(lossfun(ps_cur))
        catch e
            if e isa LinearAlgebra.PosDefException || e isa LinearAlgebra.LAPACKException
                @warn "objective rejected a step: $(typeof(e)) from the reduced overlap; Wb is near rank deficient" maxlog=3
                return 1e6
            end
            rethrow()
        end
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

# The restart protocol matters, and the default here is "do not restart".
#
# Measured on the Cu fits: restarting BFGS every 300 steps cost 21.2 meV and
# a 10x slower descent rate against a run that kept its curvature (1600 steps
# -> 205.37 meV versus 5800 -> 210.02, and later 3100 -> 191.71 versus 7700 ->
# 205.65).  The |g|_inf of the faster arm did not improve, so the gradient
# norm is not the thing to steer by.
#
# Three parts:
#   * restart_steps defaults to 10_000, i.e. >= any realistic `iterations`, so
#     there is exactly ONE Optim.optimize call and nothing is discarded;
#   * if a restart does happen it is WARM -- the previous inverse Hessian is
#     handed back through BFGS(initial_invH = ...) instead of restarting from
#     Optim's default scaling;
#   * every "no longer making progress" exit is OFF by default
#     (f_abstol = f_reltol = x_abstol = x_reltol = 0), leaving only g_tol.
#     Those tolerances are what caused the premature exits; the old hardcoded
#     f_abstol = f_reltol = 1e-10 is gone.
#
# `initial_invH` lets a CALLER carry curvature across separate calls -- the
# phased flows (multilevel levels, then Stage B / Stage C) otherwise cold-start
# BFGS four times in one run.  It is ignored with a printed note when the
# parameter dimension changed, since the matrix is then not transferable.
#
# Returns (ps, history, result, invH); the 4th element is the inverse Hessian
# to feed into the next call.
function _bfgs_refine(lossfun, ps;
    iterations::Int = 1000,
    train_part::Symbol = :all,
    g_tol::Float64 = 1e-6,
    f_abstol::Float64 = 0.0,
    f_reltol::Float64 = 0.0,
    x_abstol::Float64 = 0.0,
    x_reltol::Float64 = 0.0,
    restart_steps::Int = 10_000,
    warm_restart::Bool = true,
    initial_invH = nothing,
    precond = nothing,
    # extended_trace has to be on to recover "~inv(H)", and with the trace
    # shown Optim prints that whole matrix every iteration: a 15-iteration
    # 2-level run produced 488 KB of inverse Hessians.  Hence off by default,
    # with the per-round summary below instead.
    show_trace::Bool = false,
)
    println("    [BFGS: $iterations steps, train_part=$train_part, " *
            "restart_steps=$restart_steps" *
            (precond === nothing ? "" : ", preconditioned") *
            (initial_invH === nothing ? "" : ", warm invH") * "]")

    x0, f, rebuild_ps = _make_objective_over_part(lossfun, ps, train_part)

    Tp = eltype(first(destructure(_select_trainable_part(ps, train_part)[1])))
    if real(Tp) !== Float64
        @info "BFGS: parameters are $(real(Tp)); the objective is evaluated at " *
              "that precision, so a line search cannot resolve better than " *
              "~$(eps(real(Tp)))" maxlog=1
    end

    # Change of variables theta = P y when a preconditioner is given.  Zygote
    # differentiates through it, so the gradient the optimiser sees is P' g
    # automatically -- there is no hand-written chain to get wrong.
    if precond === nothing
        y0, F, to_theta = x0, f, identity
    else
        size(precond.P, 1) == length(x0) || error(
            "preconditioner is $(size(precond.P,1))x$(size(precond.P,2)) but " *
            "train_part=$train_part has $(length(x0)) parameters")
        y0 = precond.Pinv * x0
        F  = y -> f(precond.P * y)
        to_theta = y -> precond.P * y
    end

    function g!(storage, y)
        g_raw = Zygote.gradient(F, y)[1]
        if g_raw === nothing
            fill!(storage, 0.0)
        else
            storage .= Float64.(g_raw)
        end
        return nothing
    end

    if initial_invH !== nothing && size(initial_invH) != (length(y0), length(y0))
        println("      invH not transferable ($(size(initial_invH)) vs " *
                "$((length(y0), length(y0)))); cold start")
        initial_invH = nothing
    end

    y = copy(y0)
    history = Float64[]
    result = nothing
    invH_carry = initial_invH      # what the NEXT round starts from
    invH_out   = initial_invH      # what we hand back to the caller
    done = 0
    nround = 0   # not `round`: that would shadow Base.round

    while done < iterations
        nround += 1
        budget = min(restart_steps, iterations - done)
        bfgs = invH_carry === nothing ? Optim.BFGS() :
                   Optim.BFGS(initial_invH = _ -> copy(invH_carry))
        options = Optim.Options(
            g_tol       = g_tol,
            f_abstol    = f_abstol,
            f_reltol    = f_reltol,
            x_abstol    = x_abstol,
            x_reltol    = x_reltol,
            iterations  = budget,
            show_trace  = show_trace,
            store_trace = true,
            extended_trace = true,
        )
        result = Optim.optimize(F, g!, y, bfgs, options)
        y = copy(Optim.minimizer(result))
        append!(history, [tr.value for tr in Optim.trace(result)])
        done += Optim.iterations(result)

        # Scan backwards: the LAST trace entry does not always carry the
        # metadata (a round that exits immediately can leave only the initial
        # state), and silently returning `nothing` would turn the next call
        # into a cold start without saying so.
        for tr_e in Iterators.reverse(Optim.trace(result))
            if haskey(tr_e.metadata, "~inv(H)")
                invH_out = copy(tr_e.metadata["~inv(H)"])
                break
            end
        end

        # warm_restart = false reproduces the old cold-start behaviour for A/B.
        # It must only suppress the CARRY into the next round -- the matrix is
        # still handed back, or a cold-restart run would silently return nothing
        # and turn the caller's next call cold too.
        invH_carry = warm_restart ? invH_out : nothing

        if Optim.converged(result)
            println("      converged on round $nround after $done iterations")
            break
        end
        done < iterations && println("      " *
            (warm_restart ? "WARM" : "COLD") * " RESTART round $nround -> " *
            "$(nround+1), $done/$iterations done")
    end

    ps_cur = rebuild_ps(to_theta(y))
    println("      final loss = $(round(F(y), digits = 8))")
    return ps_cur, history, result, invH_out
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

    # Curvature hand-off across the change of objective, which is on by
    # default on measurement rather than on principle: on 2-atom Si /
    # def2-SVP, phase A on bands 1:4 then phase B on bands 1:8, 40 + 40 BFGS
    # iterations, carrying the inverse Hessian beat a cold start in 5 of 5
    # random starts, by 1.05x to 10.6x.  _bfgs_refine drops it with a printed
    # note if the parameter dimension changed.
    carry_invH::Bool       = true,
    initial_invH           = nothing,

    # Pre-flight: :error (default), :warn or :off.  An indefinite S(k) makes
    # every evaluation throw PosDefException, hence the objective Inf
    # everywhere, so this is checked before the first step rather than
    # discovered after a wasted run.  See check_overlap_pd.
    check_overlap::Symbol  = :error,
    verbose::Bool          = true,
)
    check_overlap_pd(model, data, ps, st; mode = check_overlap,
                     verbose = verbose, label = "  [train] ")

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

    ps_cur, hist2, _, invH = _bfgs_refine(
        loss_rmse, ps_cur;
        iterations = iterations,
        train_part = train_part,
        initial_invH = initial_invH,
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

        ps_cur, hist3, _, invH = _bfgs_refine(
            loss_rmse_H, ps_cur;
            iterations = H_iterations,
            train_part = H_train_part,
            initial_invH = carry_invH ? invH : nothing,
        )

        x_history = vcat(x_history, hist3)
    else
        println("    [Phase 3 skipped — ψ_ref / proj / kweights / filter not fully provided]")
    end

    return ps_cur, x_history, invH
end