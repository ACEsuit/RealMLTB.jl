using Random
using LuxCore
using LinearAlgebra
using EquivariantTensors

export generate_ladder, multilevel_train

# ============================================================
# 0) Ladder schedule
# ============================================================
function generate_ladder(target_Dtot::Int, target_maxl::Int, target_ORD::Int)
    ladder = NamedTuple[]

    start_maxl = min(3, target_maxl)
    start_Dtot = min(4, target_Dtot)

    push!(ladder, (Dtot = start_Dtot, maxl = start_maxl, ORD = 1))

    if target_ORD > 1 || target_maxl > start_maxl || target_Dtot > start_Dtot
        push!(ladder, (Dtot = target_Dtot, maxl = target_maxl, ORD = 1))
    end

    for o in 2:target_ORD
        push!(ladder, (Dtot = target_Dtot, maxl = target_maxl, ORD = o))
    end

    return unique(ladder)
end

# ============================================================
# 1) Warm-start parameter transfer utilities
# ============================================================
const NT_NL_SPEC = NamedTuple{(:n, :l), Tuple{Int, Int}}

function get_nnll_spec(tensor)
    _nl(bb) = [(n = b.n, l = b.l) for b in bb]
    spec = tensor.meta["𝔸spec"]
    NNLL = Vector{Vector{NT_NL_SPEC}}[]
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
        push!(NNLL, nnll_list)
    end
    return NNLL
end

function transfer_component_weights(tensor_small, W_small, tensor_large, W_large_template)
    specs_small = get_nnll_spec(tensor_small)
    specs_large = get_nnll_spec(tensor_large)

    W_large_new = [zeros(eltype(w), size(w)) for w in W_large_template]

    for k in 1:length(specs_large)
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
_get_WLL(ps_ace_chain)   = ps_ace_chain.ace.WLL

function _set_WLL(ps_ace_chain, WLL_new)
    ace_new = merge(ps_ace_chain.ace, (; WLL = tuple(WLL_new...)))
    return merge(ps_ace_chain, (; ace = ace_new))
end

function embed_acev_3c_params(prev_model, prev_ps, model, ps_init)
    ace_small  = prev_model.layers.layer_1.ace_model
    ace_large  = model.layers.layer_1.ace_model
    tensor_small = _get_symbasis(ace_small)
    tensor_large = _get_symbasis(ace_large)
    W_small          = _get_WLL(prev_ps.layer_1.ace)
    W_large_template = _get_WLL(ps_init.layer_1.ace)
    W_new = transfer_component_weights(tensor_small, W_small, tensor_large, W_large_template)
    ps_ace_new = _set_WLL(ps_init.layer_1.ace, W_new)
    new_layer1 = merge(ps_init.layer_1, (; ace = ps_ace_new))
    ps_new     = merge(ps_init, (; layer_1 = new_layer1))
    if hasproperty(ps_new.layer_1, :Wb) && hasproperty(prev_ps.layer_1, :Wb)
        ps_new.layer_1.Wb .= prev_ps.layer_1.Wb
    end
    return ps_new
end

function embed_ace3c_B_params(prev_model, prev_ps, model, ps_init)
    acev_small = prev_model.layers.layer_1.ace_v_model
    acev_large = model.layers.layer_1.ace_v_model
    Wv_new = transfer_component_weights(
        _get_symbasis(acev_small), _get_WLL(prev_ps.layer_1.acev),
        _get_symbasis(acev_large), _get_WLL(ps_init.layer_1.acev))
    ps_acev_new = _set_WLL(ps_init.layer_1.acev, Wv_new)
    wpsi_small = prev_model.layers.layer_1.wpsi_model
    wpsi_large = model.layers.layer_1.wpsi_model
    Wψ_new = transfer_component_weights(
        _get_symbasis(wpsi_small), _get_WLL(prev_ps.layer_1.wpsi),
        _get_symbasis(wpsi_large), _get_WLL(ps_init.layer_1.wpsi))
    ps_wpsi_new = _set_WLL(ps_init.layer_1.wpsi, Wψ_new)
    new_layer1 = merge(ps_init.layer_1, (; acev = ps_acev_new, wpsi = ps_wpsi_new))
    return merge(ps_init, (; layer_1 = new_layer1))
end

function embed_ace3c_A_params(prev_model, prev_ps, model, ps_init)
    acev_small = prev_model.layers.layer_1.ace_v_model
    acev_large = model.layers.layer_1.ace_v_model
    Wv_new = transfer_component_weights(
        _get_symbasis(acev_small), _get_WLL(prev_ps.layer_1.acev),
        _get_symbasis(acev_large), _get_WLL(ps_init.layer_1.acev))
    ps_acev_new = _set_WLL(ps_init.layer_1.acev, Wv_new)
    wpsi_small = prev_model.layers.layer_1.wpsi_model
    wpsi_large = model.layers.layer_1.wpsi_model
    Wψ_new = transfer_component_weights(
        _get_symbasis(wpsi_small), _get_WLL(prev_ps.layer_1.wpsi),
        _get_symbasis(wpsi_large), _get_WLL(ps_init.layer_1.wpsi))
    ps_wpsi_new = _set_WLL(ps_init.layer_1.wpsi, Wψ_new)
    new_layer1 = merge(ps_init.layer_1, (; acev = ps_acev_new, wpsi = ps_wpsi_new))
    return merge(ps_init, (; layer_1 = new_layer1))
end

# ============================================================
# 2) multilevel_train
# ============================================================
"""
    multilevel_train(G, λ_ref, bs, NORB, rcut; kwargs...)

Progressive λ-training across a ladder of (Dtot, maxl, ORD), with an
optional final Hamiltonian fine-tune (Phase 3) on the last level.

Extra kwargs for Phase 3:
  H_ref_AO_list  : Vector of H^proj matrices from build_ref_and_graph_lists_*
  filter         : spectral filter function, e.g. x -> 1/(1+exp((x-μ)/β))
  H_iterations   : BFGS iteration budget for Phase 3  (default 500)
  H_reg          : regularisation weight for Phase 3   (default 1e-4)

If H_ref_AO_list or filter is not provided, Phase 3 is silently skipped.

Example
-------
    fermi = x -> 1 / (1 + exp((x - μ) / β))

    ps_opt, st_opt, model = multilevel_train(
        G_list, λ_ref_list, bs_ref, NORB, rcut;
        model_type     = :ace3c_B,
        H_ref_AO_list  = H_ref_list,   # from build_ref_and_graph_lists_*
        filter         = fermi,
        H_iterations   = 500,
    )
"""
function multilevel_train(G, λ_ref, bs, NORB, rcut;
    model_type::Symbol   = :acev_3c,
    nψ                   = nothing,
    target_Dtot::Int     = 5,
    target_maxl::Int     = 5,
    target_ORD::Int      = 3,
    keep_bands::Int      = 4,
    adam_iterations      = 100,
    base_iterations::Int = 1000,
    seed::Int            = 1234,
    verbose::Bool        = true,
    embed_func           = nothing,
    λ_reg                = 1e-4,
    # --- Phase 3 (Hamiltonian fine-tune) ---
    H_ref_AO_list        = nothing,
    filter               = nothing,
    H_iterations::Int    = 500,
    H_reg                = 1e-4,
)
    rng    = MersenneTwister(seed)
    ladder = generate_ladder(target_Dtot, target_maxl, target_ORD)

    if verbose
        println("\n=== Start Progressive Training ===")
        println("Type: $model_type | target Dtot=$target_Dtot maxl=$target_maxl ORD=$target_ORD")
        println("Levels: $(length(ladder))")
        for (i, cfg) in enumerate(ladder)
            println("  Level $i: Dtot=$(cfg.Dtot), maxl=$(cfg.maxl), ORD=$(cfg.ORD)")
        end
        if H_ref_AO_list !== nothing && filter !== nothing
            println("Phase 3 (RMSE_H fine-tune): enabled on last level, $H_iterations BFGS steps")
        else
            println("Phase 3 (RMSE_H fine-tune): disabled")
        end
    end

    local model_builder

    if model_type == :simple3c
        model_builder = (bs, NORB, rcut; Dtot=0, maxl=0, ORD=0, rng=rng) ->
            simple_3c_model(bs, NORB)
        embed_func = nothing

    elseif model_type == :acev_3c
        model_builder = (bs, NORB, rcut; Dtot, maxl, ORD, rng=rng) ->
            acev_3c_model(bs, NORB, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
        embed_func === nothing && (embed_func = embed_acev_3c_params)

    elseif model_type == :ace3c_B
        model_builder = (bs, NORB, rcut; Dtot, maxl, ORD, rng=rng) ->
            ace3c_B_model(bs, NORB, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
        embed_func === nothing && (embed_func = embed_ace3c_B_params)

    elseif model_type == :ace3c_A
        nψ === nothing && error(":ace3c_A requires keyword nψ = (...)")
        model_builder = (bs, NORB, rcut; Dtot, maxl, ORD, rng=rng) ->
            ace3c_A_model(bs, nψ, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
        embed_func === nothing && (embed_func = embed_ace3c_A_params)

    else
        error("Unknown model_type: $model_type")
    end

    local prev_model    = nothing
    local prev_ps       = nothing
    local current_model = nothing
    local current_ps    = nothing
    local current_st    = nothing

    n_levels = length(ladder)

    for (i, cfg) in enumerate(ladder)
        verbose && println("\n>>> [Level $i/$n_levels] Dtot=$(cfg.Dtot) maxl=$(cfg.maxl) ORD=$(cfg.ORD)")

        current_model = model_builder(bs, NORB, rcut;
                                      Dtot=cfg.Dtot, maxl=cfg.maxl, ORD=cfg.ORD, rng=rng)
        ps_init, st_init = LuxCore.setup(rng, current_model)
        current_st = st_init

        if i == 1 || embed_func === nothing
            current_ps = ps_init
            verbose && println("    Init: random")
        else
            verbose && println("    Init: warm-start from Level $(i-1)")
            current_ps = embed_func(prev_model, prev_ps, current_model, ps_init)
            if verbose
                err0 = RMSE(G, λ_ref, current_model, current_ps, current_st;
                            keep_bands=keep_bands, λ_reg=λ_reg)
                println("    RMSE after warm-start: $err0")
            end
        end

        # Phase 3 only fires on the last ladder level
        is_last_level = (i == n_levels)
        level_H_ref   = is_last_level ? H_ref_AO_list : nothing
        level_filter  = is_last_level ? filter         : nothing

        verbose && println("    Training: $base_iterations iterations")
        ps_opt, _ = train(G, current_model, current_ps, current_st, keep_bands, λ_ref;
                          adam_iterations = adam_iterations,
                          iterations      = base_iterations,
                          λ_reg           = λ_reg,
                          H_ref_AO_list   = level_H_ref,
                          filter          = level_filter,
                          H_iterations    = H_iterations,
                          H_reg           = H_reg)
        current_ps = ps_opt

        if verbose
            err1 = RMSE(G, λ_ref, current_model, current_ps, current_st;
                        keep_bands=keep_bands, λ_reg=λ_reg)
            println("    Done Level $i. RMSE_λ: $err1")
            if is_last_level && H_ref_AO_list !== nothing && filter !== nothing
                err_H = RMSE_H(G, H_ref_AO_list, current_model, current_ps, current_st, filter)
                println("    Done Level $i. RMSE_H: $err_H")
            end
        end

        prev_model = current_model
        prev_ps    = current_ps
    end

    verbose && println("\n=== Progressive Training Completed ===")
    return current_ps, current_st, current_model
end


function multilevel_train_H(G, bs, NORB, rcut;
    H_ref_AO_list,     
    filter,           
    model_type::Symbol   = :acev_3c,
    nψ                   = nothing,
    target_Dtot::Int     = 5,
    target_maxl::Int     = 5,
    target_ORD::Int      = 3,
    H_iterations::Int    = 500,
    H_reg                = 1e-4,
    seed::Int            = 1234,
    verbose::Bool        = true,
    embed_func           = nothing,
)

    λ_ref_dummy = [
        [zeros(1) for _ in 1:length(g.graph_data.klist)]
        for g in (G isa AbstractVector ? G : [G])
    ]
    G_input = G isa AbstractVector ? G : [G]

    return multilevel_train(
        G_input, λ_ref_dummy, bs, NORB, rcut;
        model_type      = model_type,
        nψ              = nψ,
        target_Dtot     = target_Dtot,
        target_maxl     = target_maxl,
        target_ORD      = target_ORD,
        keep_bands      = 1,     
        adam_iterations = 0,     
        base_iterations = 0,  
        seed            = seed,
        verbose         = verbose,
        embed_func      = embed_func,
        H_ref_AO_list   = H_ref_AO_list,
        filter          = filter,
        H_iterations    = H_iterations,
        H_reg           = H_reg,
    )
end