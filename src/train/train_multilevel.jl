using Random
using LinearAlgebra
using LuxCore
using EquivariantTensors

export generate_ladder, multilevel_train, multilevel_train_H

# ============================================================================
# Ladder schedule
# ============================================================================

function generate_ladder(target_Dtot::Int, target_maxl::Int, target_ORD::Int; steps::Int = 5)
    ladder = NamedTuple[]

    start_maxl = min(3, target_maxl)
    start_Dtot = min(4, target_Dtot)

    d_vals = round.(Int, range(start_Dtot, target_Dtot, length = steps))
    l_vals = round.(Int, range(start_maxl, target_maxl, length = steps))

    for (d, l) in zip(d_vals, l_vals)
        push!(ladder, (Dtot = d, maxl = l, ORD = 1))
    end

    if target_ORD >= 2
        for o in 2:target_ORD
            push!(ladder, (Dtot = target_Dtot, maxl = target_maxl, ORD = o))
        end
    end

    return unique(ladder)
end

# ============================================================================
# Warm-start parameter transfer utilities
# ============================================================================

const NT_NL_SPEC = NamedTuple{(:n, :l), Tuple{Int, Int}}

function get_nnll_spec(tensor)
    _nl(bb) = [(n = b.n, l = b.l) for b in bb]
    spec = tensor.meta["𝔸spec"]
    nnll_all = Vector{Vector{NT_NL_SPEC}}[]

    for A2Bmap in tensor.A2Bmaps
        nBB = size(A2Bmap, 1)
        nnll_list = Vector{NT_NL_SPEC}[]

        for i in 1:nBB
            AAidx_nnz = A2Bmap[i, :].nzind
            bbs = spec[AAidx_nnz]
            signature = _nl(bbs[1])

            @assert all(signature == _nl(bb) for bb in bbs)
            push!(nnll_list, signature)
        end

        push!(nnll_all, nnll_list)
    end

    return nnll_all
end

function transfer_component_weights(tensor_small, W_small, tensor_large, W_large_template)
    specs_small = get_nnll_spec(tensor_small)
    specs_large = get_nnll_spec(tensor_large)

    # Preserve the storage backend of W_large_template.
    # If W_large_template is on GPU, similar(w) also allocates on GPU.
    W_large_new = map(W_large_template) do w
        z = similar(w)
        fill!(z, zero(eltype(z)))
        z
    end

    for k in eachindex(specs_large)
        small_lookup = Dict{Vector{NT_NL_SPEC}, Int}()

        for (idx, spec) in enumerate(specs_small[k])
            small_lookup[spec] = idx
        end

        n_rows_large, _ = size(W_large_new[k])

        for idx_L in 1:n_rows_large
            spec_L = specs_large[k][idx_L]

            if haskey(small_lookup, spec_L)
                idx_S = small_lookup[spec_L]
                W_large_new[k][idx_L, :] = W_small[k][idx_S, :]
            end
        end
    end

    return W_large_new
end

_get_symbasis(ace_chain) = ace_chain.layers.ace.symbasis
_get_WLL(ps_ace_chain) = ps_ace_chain.ace.WLL

function _set_WLL(ps_ace_chain, WLL_new)
    ace_new = merge(ps_ace_chain.ace, (; WLL = tuple(WLL_new...)))
    return merge(ps_ace_chain, (; ace = ace_new))
end

function embed_acev_3c_params(prev_model, prev_ps, model, ps_init)
    ace_small = prev_model.layers.layer_1.ace_model
    ace_large = model.layers.layer_1.ace_model

    tensor_small = _get_symbasis(ace_small)
    tensor_large = _get_symbasis(ace_large)

    W_small = _get_WLL(prev_ps.layer_1.ace)
    W_large_template = _get_WLL(ps_init.layer_1.ace)

    W_new = transfer_component_weights(tensor_small, W_small, tensor_large, W_large_template)

    ps_ace_new = _set_WLL(ps_init.layer_1.ace, W_new)
    new_layer1 = merge(ps_init.layer_1, (; ace = ps_ace_new))
    ps_new = merge(ps_init, (; layer_1 = new_layer1))

    if hasproperty(ps_new.layer_1, :Wb) && hasproperty(prev_ps.layer_1, :Wb)
        ps_new.layer_1.Wb .= prev_ps.layer_1.Wb
    end

    return ps_new
end

function _embed_ace3c_params(prev_model, prev_ps, model, ps_init)
    acev_small = prev_model.layers.layer_1.ace_v_model
    acev_large = model.layers.layer_1.ace_v_model

    Wv_new = transfer_component_weights(
        _get_symbasis(acev_small),
        _get_WLL(prev_ps.layer_1.acev),
        _get_symbasis(acev_large),
        _get_WLL(ps_init.layer_1.acev),
    )

    ps_acev_new = _set_WLL(ps_init.layer_1.acev, Wv_new)

    wpsi_small = prev_model.layers.layer_1.wpsi_model
    wpsi_large = model.layers.layer_1.wpsi_model

    Wψ_new = transfer_component_weights(
        _get_symbasis(wpsi_small),
        _get_WLL(prev_ps.layer_1.wpsi),
        _get_symbasis(wpsi_large),
        _get_WLL(ps_init.layer_1.wpsi),
    )

    ps_wpsi_new = _set_WLL(ps_init.layer_1.wpsi, Wψ_new)

    new_layer1 = merge(ps_init.layer_1, (; acev = ps_acev_new, wpsi = ps_wpsi_new))

    return merge(ps_init, (; layer_1 = new_layer1))
end

embed_ace3c_B_params(prev_model, prev_ps, model, ps_init) =
    _embed_ace3c_params(prev_model, prev_ps, model, ps_init)

embed_ace3c_A_params(prev_model, prev_ps, model, ps_init) =
    _embed_ace3c_params(prev_model, prev_ps, model, ps_init)

# ============================================================================
# Model / embedding dispatch
# ============================================================================

function _build_model_builder(model_type::Symbol, nψ, rng, embed_func)
    if model_type == :simple3c
        model_builder = (bs, NORB, rcut; Dtot = 0, maxl = 0, ORD = 0, rng = rng) ->
            simple_3c_model(bs, NORB)

        return model_builder, nothing

    elseif model_type == :acev_3c
        model_builder = (bs, NORB, rcut; Dtot, maxl, ORD, rng = rng) ->
            acev_3c_model(bs, NORB, rcut; Dtot = Dtot, maxl = maxl, ORD = ORD, rng = rng)

        embed = isnothing(embed_func) ? embed_acev_3c_params : embed_func

        return model_builder, embed

    elseif model_type == :ace3c_B
        model_builder = (bs, NORB, rcut; Dtot, maxl, ORD, rng = rng) ->
            ace3c_B_model(bs, NORB, rcut; Dtot = Dtot, maxl = maxl, ORD = ORD, rng = rng)

        embed = isnothing(embed_func) ? embed_ace3c_B_params : embed_func

        return model_builder, embed

    elseif model_type == :ace3c_A
        isnothing(nψ) && error(":ace3c_A requires keyword nψ = (...)")

        model_builder = (bs, NORB, rcut; Dtot, maxl, ORD, rng = rng) ->
            ace3c_A_model(bs, nψ, rcut; Dtot = Dtot, maxl = maxl, ORD = ORD, rng = rng)

        embed = isnothing(embed_func) ? embed_ace3c_A_params : embed_func

        return model_builder, embed

    else
        error("Unknown model_type: $model_type")
    end
end

# ============================================================================
# Internal helpers
# ============================================================================

function _phase3_enabled(ψ_ref, proj, kweights, filter)
    return ψ_ref !== nothing &&
           proj !== nothing &&
           kweights !== nothing &&
           filter !== nothing
end

function _print_ladder_summary(model_type, target_Dtot, target_maxl, target_ORD,
                               ladder, ψ_ref, proj, kweights, filter, H_iterations;
                               train_part::Symbol = :all,
                               H_train_part::Symbol = train_part)
    println("\n=== Start Progressive Training ===")
    println("Type: $model_type | target Dtot=$target_Dtot maxl=$target_maxl ORD=$target_ORD")
    println("Levels: $(length(ladder))")
    println("Train part: $train_part | H train part: $H_train_part")

    for (i, cfg) in enumerate(ladder)
        println("  Level $i: Dtot=$(cfg.Dtot), maxl=$(cfg.maxl), ORD=$(cfg.ORD)")
    end

    if _phase3_enabled(ψ_ref, proj, kweights, filter)
        println("Phase 3 (RMSE_H fine-tune): enabled, $H_iterations BFGS steps")
    else
        println("Phase 3 (RMSE_H fine-tune): disabled")
    end
end

function _initialise_level(
    model_builder,
    bs,
    NORB,
    rcut,
    cfg,
    rng;
    prepare_ps_st = (ps, st) -> (ps, st),
)
    model = model_builder(
        bs,
        NORB,
        rcut;
        Dtot = cfg.Dtot,
        maxl = cfg.maxl,
        ORD = cfg.ORD,
        rng = rng,
    )

    ps_init, st_init = LuxCore.setup(rng, model)
    ps_init, st_init = prepare_ps_st(ps_init, st_init)

    return model, ps_init, st_init
end

function _prepare_level_parameters(i, verbose, prev_model, prev_ps, current_model,
                                   ps_init, st_init, embed_func, G, λ_ref,
                                   keep_bands, ψ_ref, proj, kweights, filter)
    if i == 1 || isnothing(embed_func)
        verbose && println("    Init: random")
        return ps_init, st_init
    end

    verbose && println("    Init: warm-start from Level $(i - 1)")

    ps_warm = embed_func(prev_model, prev_ps, current_model, ps_init)

    if verbose
        if _phase3_enabled(ψ_ref, proj, kweights, filter)
            err0 = RMSE_H(
                G,
                λ_ref,
                ψ_ref,
                proj,
                kweights,
                filter,
                current_model,
                ps_warm,
                st_init,
            )

            println("    RMSE_H after warm-start: $err0")
        else
            err0 = RMSE(
                G,
                λ_ref,
                current_model,
                ps_warm,
                st_init;
                keep_bands = keep_bands,
            )

            println("    RMSE after warm-start: $err0")
        end
    end

    return ps_warm, st_init
end

function _train_level(G, λ_ref, model, ps, st, keep_bands;
    adam_iterations,
    base_iterations,
    adam_lr,
    grad_clip,
    adam_patience,
    ψ_ref,
    proj,
    kweights,
    filter,
    H_iterations,
    train_part::Symbol = :all,
    H_train_part::Symbol = train_part,
    initial_invH = nothing,
    check_overlap::Symbol = :error,
)
    ps_opt, history, invH = train(
        G,
        model,
        ps,
        st,
        keep_bands,
        λ_ref;
        iterations = base_iterations,
        adam_iterations = adam_iterations,
        adam_lr = adam_lr,
        grad_clip = grad_clip,
        adam_patience = adam_patience,
        train_part = train_part,
        H_train_part = H_train_part,
        ψ_ref = ψ_ref,
        proj = proj,
        kweights = kweights,
        filter = filter,
        H_iterations = H_iterations,
        initial_invH = initial_invH,
        check_overlap = check_overlap,
    )

    return ps_opt, history, invH
end

function _train_level_H(G, λ_ref, ψ_ref, proj, kweights, filter, model, ps, st;
    H_iterations::Int,
    cone_w::Float64,
    filter_σ::Float64,
    train_part::Symbol = :all,
    initial_invH = nothing,
    check_overlap::Symbol = :error,
)
    # this path calls _bfgs_refine directly rather than going through train(),
    # so it needs its own pre-flight
    check_overlap_pd(model, G, ps, st; mode = check_overlap,
                     label = "  [_train_level_H] ")

    loss_H(p) = RMSE_H(
        G,
        λ_ref,
        ψ_ref,
        proj,
        kweights,
        filter,
        model,
        p,
        st;
        cone_w = cone_w,
        filter_σ = filter_σ,
    )

    ps_opt, history, _, invH = _bfgs_refine(
        loss_H,
        ps;
        iterations = H_iterations,
        train_part = train_part,
        initial_invH = initial_invH,
    )

    return ps_opt, history, invH
end

function _report_level_done(verbose, i, G, λ_ref, model, ps, st, keep_bands,
                            ψ_ref, proj, kweights, filter, cone_w, filter_σ)
    verbose || return

    if _phase3_enabled(ψ_ref, proj, kweights, filter)
        err_H = RMSE_H(
            G,
            λ_ref,
            ψ_ref,
            proj,
            kweights,
            filter,
            model,
            ps,
            st;
            cone_w = cone_w,
            filter_σ = filter_σ,
        )

        println("    Done Level $i. RMSE_H: $err_H")
    else
        err1 = RMSE(
            G,
            λ_ref,
            model,
            ps,
            st;
            keep_bands = keep_bands,
        )

        println("    Done Level $i. RMSE_λ: $err1")
    end
end

# ============================================================================
# multilevel_train
# ============================================================================

function multilevel_train(G, λ_ref, bs, NORB, rcut;
    model_type::Symbol = :acev_3c,
    nψ = nothing,
    target_Dtot::Int = 5,
    target_maxl::Int = 5,
    target_ORD::Int = 3,
    keep_bands::Int = 4,
    adam_iterations::Int = 100,
    base_iterations::Int = 1000,
    adam_lr::Float64 = 1e-3,
    grad_clip::Float64 = 10.0,
    adam_patience::Int = max(adam_iterations, 1),
    verbose::Bool = true,
    embed_func = nothing,

    # New: choose which part to train.
    #   :all
    #   :potential
    #   :basis
    train_part::Symbol = :all,
    H_train_part::Symbol = train_part,

    ψ_ref = nothing,
    proj = nothing,
    kweights = nothing,
    filter = nothing,
    H_iterations::Int = 500,
    prepare_ps_st = (ps, st) -> (ps, st),

    # pre-flight overlap check, forwarded to every level: :error, :warn, :off
    check_overlap::Symbol = :error,
)
    rng = MersenneTwister()
    ladder = generate_ladder(target_Dtot, target_maxl, target_ORD)

    verbose && _print_ladder_summary(
        model_type,
        target_Dtot,
        target_maxl,
        target_ORD,
        ladder,
        ψ_ref,
        proj,
        kweights,
        filter,
        H_iterations;
        train_part = train_part,
        H_train_part = H_train_part,
    )

    model_builder, embed_func = _build_model_builder(model_type, nψ, rng, embed_func)

    prev_model = nothing
    prev_ps = nothing

    current_model = nothing
    current_ps = nothing
    current_st = nothing

    # Curvature carried from the previous level.  The ladder changes the
    # parameter dimension whenever Dtot/maxl/ORD change, and _bfgs_refine then
    # drops it with a printed note -- so this pays at the level boundaries
    # where the dimension happens to be unchanged, and is a no-op elsewhere.
    level_invH = nothing

    n_levels = length(ladder)

    for (i, cfg) in enumerate(ladder)
        verbose && println("\n>>> [Level $i/$n_levels] Dtot=$(cfg.Dtot) maxl=$(cfg.maxl) ORD=$(cfg.ORD)")

        current_model, ps_init, st_init =
            _initialise_level(
                model_builder,
                bs,
                NORB,
                rcut,
                cfg,
                rng;
                prepare_ps_st = prepare_ps_st,
            )

        current_ps, current_st = _prepare_level_parameters(
            i,
            verbose,
            prev_model,
            prev_ps,
            current_model,
            ps_init,
            st_init,
            embed_func,
            G,
            λ_ref,
            keep_bands,
            ψ_ref,
            proj,
            kweights,
            filter,
        )

        verbose && println(
            "    Training: $base_iterations BFGS iterations, " *
            "$adam_iterations Adam steps, train_part=$train_part, H_train_part=$H_train_part"
        )

        current_ps, _, level_invH = _train_level(
            G,
            λ_ref,
            current_model,
            current_ps,
            current_st,
            keep_bands;
            adam_iterations = adam_iterations,
            base_iterations = base_iterations,
            adam_lr = adam_lr,
            grad_clip = grad_clip,
            adam_patience = adam_patience,
            check_overlap = check_overlap,
            ψ_ref = ψ_ref,
            proj = proj,
            kweights = kweights,
            filter = filter,
            H_iterations = H_iterations,
            train_part = train_part,
            H_train_part = H_train_part,
            initial_invH = level_invH,
        )

        _report_level_done(
            verbose,
            i,
            G,
            λ_ref,
            current_model,
            current_ps,
            current_st,
            keep_bands,
            ψ_ref,
            proj,
            kweights,
            filter,
            1.0,
            1.0,
        )

        prev_model = current_model
        prev_ps = current_ps
    end

    verbose && println("\n=== Progressive Training Completed ===")

    return current_ps, current_st, current_model
end

# ============================================================================
# Hamiltonian-only multilevel training
# ============================================================================

function multilevel_train_H(G, λ_ref, ψ_ref, proj, kweights, bs, NORB, rcut;
    filter,
    model_type::Symbol = :acev_3c,
    nψ = nothing,
    target_Dtot::Int = 5,
    target_maxl::Int = 5,
    target_ORD::Int = 3,
    H_iterations::Int = 500,
    cone_w::Float64 = 1.0,
    filter_σ::Float64 = 0.5,
    verbose::Bool = true,
    embed_func = nothing,
    prepare_ps_st = (ps, st) -> (ps, st),

    # New:
    #   :all
    #   :potential
    #   :basis
    train_part::Symbol = :all,

    # pre-flight overlap check, forwarded to every level: :error, :warn, :off
    check_overlap::Symbol = :error,
)
    rng = MersenneTwister()
    ladder = generate_ladder(target_Dtot, target_maxl, target_ORD)

    if verbose
        println("\n=== Start H-only Progressive Training ===")
        println("Type: $model_type | target Dtot=$target_Dtot maxl=$target_maxl ORD=$target_ORD")
        println("Levels: $(length(ladder))")
        println("H-only train_part: $train_part")

        for (i, cfg) in enumerate(ladder)
            println("  Level $i: Dtot=$(cfg.Dtot), maxl=$(cfg.maxl), ORD=$(cfg.ORD)")
        end

        println("H-only training: every level uses RMSE_H, $H_iterations BFGS steps")
    end

    model_builder, embed_func = _build_model_builder(model_type, nψ, rng, embed_func)

    prev_model = nothing
    prev_ps = nothing

    current_model = nothing
    current_ps = nothing
    current_st = nothing

    # Curvature carried from the previous level.  The ladder changes the
    # parameter dimension whenever Dtot/maxl/ORD change, and _bfgs_refine then
    # drops it with a printed note -- so this pays at the level boundaries
    # where the dimension happens to be unchanged, and is a no-op elsewhere.
    level_invH = nothing

    n_levels = length(ladder)

    for (i, cfg) in enumerate(ladder)
        verbose && println("\n>>> [Level $i/$n_levels] Dtot=$(cfg.Dtot) maxl=$(cfg.maxl) ORD=$(cfg.ORD)")

        current_model, ps_init, st_init =
            _initialise_level(
                model_builder,
                bs,
                NORB,
                rcut,
                cfg,
                rng;
                prepare_ps_st = prepare_ps_st,
            )

        current_ps, current_st = _prepare_level_parameters(
            i,
            verbose,
            prev_model,
            prev_ps,
            current_model,
            ps_init,
            st_init,
            embed_func,
            G,
            λ_ref,
            1,
            ψ_ref,
            proj,
            kweights,
            filter,
        )

        verbose && println("    Training: $H_iterations BFGS steps on RMSE_H, train_part=$train_part")

        current_ps, _, level_invH = _train_level_H(
            G,
            λ_ref,
            ψ_ref,
            proj,
            kweights,
            filter,
            current_model,
            current_ps,
            current_st;
            H_iterations = H_iterations,
            cone_w = cone_w,
            filter_σ = filter_σ,
            train_part = train_part,
            initial_invH = level_invH,
            check_overlap = check_overlap,
        )

        _report_level_done(
            verbose,
            i,
            G,
            λ_ref,
            current_model,
            current_ps,
            current_st,
            1,
            ψ_ref,
            proj,
            kweights,
            filter,
            cone_w,
            filter_σ,
        )

        prev_model = current_model
        prev_ps = current_ps
    end

    verbose && println("\n=== H-only Progressive Training Completed ===")

    return current_ps, current_st, current_model
end