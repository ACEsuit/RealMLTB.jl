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
# 1) Warm-start embedding utilities (WLL transfer by NNLL signature)
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

# ---- helpers for ACE Lux.Chain(; embed=..., ace=...) ----
_get_symbasis(ace_chain) = ace_chain.layers.ace.symbasis
_get_WLL(ps_ace_chain)   = ps_ace_chain.ace.WLL

function _set_WLL(ps_ace_chain, WLL_new)
    ace_new = merge(ps_ace_chain.ace, (; WLL = tuple(WLL_new...)))
    return merge(ps_ace_chain, (; ace = ace_new))
end

# ---- embed for acev_3c_model ----
function embed_acev_3c_params(prev_model, prev_ps, model, ps_init)
    ace_small = prev_model.layers.layer_1.ace_model
    ace_large = model.layers.layer_1.ace_model

    tensor_small = _get_symbasis(ace_small)
    tensor_large = _get_symbasis(ace_large)

    W_small          = _get_WLL(prev_ps.layer_1.ace)
    W_large_template = _get_WLL(ps_init.layer_1.ace)
    W_new = transfer_component_weights(tensor_small, W_small, tensor_large, W_large_template)

    ps_ace_new = _set_WLL(ps_init.layer_1.ace, W_new)

    new_layer1 = merge(ps_init.layer_1, (; ace = ps_ace_new))
    ps_new     = merge(ps_init, (; layer_1 = new_layer1))

    # also carry Wb if exists
    if hasproperty(ps_new.layer_1, :Wb) && hasproperty(prev_ps.layer_1, :Wb)
        ps_new.layer_1.Wb .= prev_ps.layer_1.Wb
    end
    return ps_new
end

# ---- embed for ace3c_B_model (CaseB Wpsi) ----
function embed_ace3c_B_params(prev_model, prev_ps, model, ps_init)
    # ace-v
    acev_small = prev_model.layers.layer_1.ace_v_model
    acev_large = model.layers.layer_1.ace_v_model
    tensor_v_small = _get_symbasis(acev_small)
    tensor_v_large = _get_symbasis(acev_large)

    Wv_small          = _get_WLL(prev_ps.layer_1.acev)
    Wv_large_template = _get_WLL(ps_init.layer_1.acev)
    Wv_new = transfer_component_weights(tensor_v_small, Wv_small, tensor_v_large, Wv_large_template)
    ps_acev_new = _set_WLL(ps_init.layer_1.acev, Wv_new)

    # wpsi (CaseB)
    wpsi_small = prev_model.layers.layer_1.wpsi_model
    wpsi_large = model.layers.layer_1.wpsi_model
    tensor_ψ_small = _get_symbasis(wpsi_small)
    tensor_ψ_large = _get_symbasis(wpsi_large)

    Wψ_small          = _get_WLL(prev_ps.layer_1.wpsi)
    Wψ_large_template = _get_WLL(ps_init.layer_1.wpsi)
    Wψ_new = transfer_component_weights(tensor_ψ_small, Wψ_small, tensor_ψ_large, Wψ_large_template)
    ps_wpsi_new = _set_WLL(ps_init.layer_1.wpsi, Wψ_new)

    new_layer1 = merge(ps_init.layer_1, (; acev = ps_acev_new, wpsi = ps_wpsi_new))
    return merge(ps_init, (; layer_1 = new_layer1))
end

# ---- embed for ace3c_A_model (CaseA Wpsi) ----
function embed_ace3c_A_params(prev_model, prev_ps, model, ps_init)
    # ace-v
    acev_small = prev_model.layers.layer_1.ace_v_model
    acev_large = model.layers.layer_1.ace_v_model
    tensor_v_small = _get_symbasis(acev_small)
    tensor_v_large = _get_symbasis(acev_large)

    Wv_small          = _get_WLL(prev_ps.layer_1.acev)
    Wv_large_template = _get_WLL(ps_init.layer_1.acev)
    Wv_new = transfer_component_weights(tensor_v_small, Wv_small, tensor_v_large, Wv_large_template)
    ps_acev_new = _set_WLL(ps_init.layer_1.acev, Wv_new)

    # wpsi (CaseA invariant scalar ACE)
    wpsi_small = prev_model.layers.layer_1.wpsi_model
    wpsi_large = model.layers.layer_1.wpsi_model
    tensor_ψ_small = _get_symbasis(wpsi_small)
    tensor_ψ_large = _get_symbasis(wpsi_large)

    Wψ_small          = _get_WLL(prev_ps.layer_1.wpsi)
    Wψ_large_template = _get_WLL(ps_init.layer_1.wpsi)
    Wψ_new = transfer_component_weights(tensor_ψ_small, Wψ_small, tensor_ψ_large, Wψ_large_template)
    ps_wpsi_new = _set_WLL(ps_init.layer_1.wpsi, Wψ_new)

    new_layer1 = merge(ps_init.layer_1, (; acev = ps_acev_new, wpsi = ps_wpsi_new))
    return merge(ps_init, (; layer_1 = new_layer1))
end

# ============================================================
# 2) multilevel_train
# ============================================================
"""
    multilevel_train(G, λ_ref, bs, NORB, rcut; model_type=:acev_3c, ...)

Progressive training across a ladder of (Dtot,maxl,ORD) with optional warm-start.

Supported model_type:
  :simple3c
  :acev_3c
  :ace3c_B   (Wpsi CaseB)
  :ace3c_A   (Wpsi CaseA; requires nψ keyword)
"""
function multilevel_train(G, λ_ref, bs, NORB, rcut;
    model_type::Symbol = :acev_3c,
    nψ = nothing,                      # required only for :ace3c_A
    target_Dtot::Int = 5,
    target_maxl::Int = 5,
    target_ORD::Int = 3,
    keep_bands::Int = 4,
    base_iterations::Int = 1000,
    seed::Int = 1234,
    verbose::Bool = true,
    embed_func = nothing,              # you can override; default auto-selected
)

    rng = MersenneTwister(seed)
    ladder = generate_ladder(target_Dtot, target_maxl, target_ORD)

    if verbose
        println("\n=== Start Progressive Training ===")
        println("Type: $model_type | target Dtot=$target_Dtot maxl=$target_maxl ORD=$target_ORD")
        println("Levels: $(length(ladder))")
        for (i, cfg) in enumerate(ladder)
            println("  Level $i: Dtot=$(cfg.Dtot), maxl=$(cfg.maxl), ORD=$(cfg.ORD)")
        end
    end

    # ---- choose model builder + default embed ----
    model_builder = nothing

    if model_type == :simple3c
        model_builder = (bs, NORB, rcut; Dtot=0, maxl=0, ORD=0, rng=rng) -> simple_3c_model(bs, NORB)
        embed_func = nothing

    elseif model_type == :acev_3c
        model_builder = (bs, NORB, rcut; Dtot, maxl, ORD, rng=rng) -> acev_3c_model(bs, NORB, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
        embed_func === nothing && (embed_func = embed_acev_3c_params)

    elseif model_type == :ace3c_B
        model_builder = (bs, NORB, rcut; Dtot, maxl, ORD, rng=rng) -> ace3c_B_model(bs, NORB, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
        embed_func === nothing && (embed_func = embed_ace3c_B_params)

    elseif model_type == :ace3c_A
        nψ === nothing && error(":ace3c_A needs keyword nψ = (..)")
        model_builder = (bs, NORB, rcut; Dtot, maxl, ORD, rng=rng) -> ace3c_A_model(bs, nψ, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
        embed_func === nothing && (embed_func = embed_ace3c_A_params)

    else
        error("Unknown model_type: $model_type")
    end

    local prev_model = nothing
    local prev_ps    = nothing

    local current_model = nothing
    local current_ps    = nothing
    local current_st    = nothing

    for (i, cfg) in enumerate(ladder)
        if verbose
            println("\n>>> [Level $i/$(length(ladder))] build: Dtot=$(cfg.Dtot) maxl=$(cfg.maxl) ORD=$(cfg.ORD)")
        end

        current_model = model_builder(bs, NORB, rcut; Dtot=cfg.Dtot, maxl=cfg.maxl, ORD=cfg.ORD, rng=rng)
        ps_init, st_init = LuxCore.setup(rng, current_model)
        current_st = st_init

        # init ps for training
        if i == 1 || embed_func === nothing
            current_ps = ps_init
            verbose && println("    Init: random")
        else
            verbose && println("    Init: embed from Level $(i-1)")
            current_ps = embed_func(prev_model, prev_ps, current_model, ps_init)

            # optional sanity check
            if verbose
                err0 = RMSE(G, λ_ref, current_model, current_ps, current_st; keep_bands=keep_bands)
                println("    (check) RMSE after embed: $err0")
            end
        end

        verbose && println("    Train: $base_iterations iters")
        ps_opt, _ = train(G, current_model, current_ps, current_st, keep_bands, λ_ref; iterations=base_iterations)
        current_ps = ps_opt

        if verbose
            err1 = RMSE(G, λ_ref, current_model, current_ps, current_st; keep_bands=keep_bands)
            println("    Done Level $i. RMSE: $err1")
        end

        prev_model = current_model
        prev_ps    = current_ps
    end

    verbose && println("\n=== Progressive Training Completed ===")
    return current_ps, current_st, current_model
end
