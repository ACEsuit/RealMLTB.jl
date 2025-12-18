export RMSE, train, multilevel_train

# --- single-sample RMSE ---
function RMSE(G, λ_ref, model, ps, st; keep_bands = 4)
    λ_pred = model(G, ps, st)[1]
    λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))
    return sqrt(norm(λ_pred_keep .- λ_ref)^2/length(λ_pred))
end

# --- multi-sample RMSE over a list ---
function RMSE(Glist::AbstractVector, λ_ref_list::AbstractVector,
              model, ps, st; keep_bands = 4)
    @assert length(Glist) == length(λ_ref_list)

    sumsq = 0.0
    n     = 0

    for (G, λ_ref) in zip(Glist, λ_ref_list)
        λ_pred = model(G, ps, st)[1]
        λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))

        diff = λ_pred_keep .- λ_ref
        sumsq += norm(diff)^2
        n     += length(diff)
    end

    return sqrt(sumsq / n)
end

# --- train for single sample ---
function train(G, model, ps, st, keep_bands, λ_ref; iterations = 1000)
    x0, re = destructure(ps)
    f(x) = RMSE(G, λ_ref, model, re(x), st; keep_bands = keep_bands)

    function g!(storage, x)
        storage .= Zygote.gradient(xx -> f(xx), x)[1]
    end

    options = Optim.Options(
        g_tol = 1e-8, f_abstol = 1e-10, f_reltol = 1e-10,
        iterations = iterations, show_trace = true, store_trace = true
    )
    result = optimize(f, g!, x0, BFGS(), options)

    trace = result.trace
    x_history = [trace[i].value for i in 1:length(trace)]

    x_opt = Optim.minimizer(result)
    ps = re(x_opt)
    return ps, x_history
end

function generate_ladder(target_Dtot, target_maxl, target_ORD)
    ladder = []
    
    start_maxl = min(3, target_maxl)
    start_Dtot = min(4, target_Dtot)
    push!(ladder, (Dtot=start_Dtot, maxl=start_maxl, ORD=1))
    
    if target_ORD > 1 || target_maxl > start_maxl
        push!(ladder, (Dtot=target_Dtot, maxl=target_maxl, ORD=1))
    end
    
    for o in 2:target_ORD
        push!(ladder, (Dtot=target_Dtot, maxl=target_maxl, ORD=o))
    end
    
    return unique(ladder)
end

function multilevel_train(G, λ_ref, bs_sph, NORB, rcut;
                           model_type = :acev_3c,
                           target_Dtot = 3,
                           target_maxl = 5,
                           target_ORD = 2,
                           keep_bands = 4,
                           base_iterations = 1000,
                           seed = 1234,
                           verbose = true)

    rng = MersenneTwister(seed)
    
    ladder = generate_ladder(target_Dtot, target_maxl, target_ORD)
    
    if verbose
        println("\n=== Start Progressive Training ===")
        println("Target: $model_type | Dtot=$target_Dtot, maxl=$target_maxl, ORD=$target_ORD")
        println("Schedule: $(length(ladder)) Levels")
        for (i, cfg) in enumerate(ladder)
            println("  Level $i: ORD=$(cfg.ORD), maxl=$(cfg.maxl), Dtot=$(cfg.Dtot)")
        end
    end

    if model_type == :acev_3c
        model_builder = acev_3c_model
        embed_func    = embed_acev_3c_params
    elseif model_type == :ace_3c
        model_builder = ace_3c_model
        embed_func    = embed_ace_3c_params
    else
        error("Unknown type: $model_type")
    end

    local prev_model = nothing
    local prev_ps = nothing
    local current_model = nothing
    local current_ps = nothing
    local current_st = nothing

    for (i, cfg) in enumerate(ladder)
        if verbose; println("\n>>> [Level $i / $(length(ladder))] Building model: ORD=$(cfg.ORD), maxl=$(cfg.maxl)..."); end
        
        current_model = model_builder(bs_sph, NORB, rcut; Dtot=cfg.Dtot, maxl=cfg.maxl, ORD=cfg.ORD)
        ps_init, st = LuxCore.setup(rng, current_model)
        current_st = st
        
        if i == 1
            if verbose; println("    Init: Random (Cold Start)"); end
            ps_train_start = ps_init
        else
            if verbose; println("    Init: Embedding from Level $(i-1)..."); end
            ps_train_start = embed_func(prev_model, prev_ps, current_model, ps_init)
            
            err_start = RMSE(G, λ_ref, current_model, ps_train_start, current_st; keep_bands = keep_bands)
            if verbose; println("    (Check) Initial RMSE after embed: $err_start"); end
        end

        if verbose; println("    Start training ($base_iterations steps)..."); end
        ps_opt, _ = train(G, current_model, ps_train_start, current_st, keep_bands, λ_ref; iterations = base_iterations)

        err_end = RMSE(G, λ_ref, current_model, ps_opt, current_st; keep_bands = keep_bands)
        if verbose; println("    >>> Level $i Done. RMSE: $err_end"); end

        prev_model = current_model
        prev_ps = ps_opt
        current_ps = ps_opt
    end

    if verbose; println("\n=== Progressive Training Completed ==="); end
    return current_ps, current_st, current_model
end

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
            @assert all([signature == _nl(bb) for bb in bbs])
            
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
        n_rows_large, n_cols = size(W_large_new[k])
        
        small_lookup = Dict{Vector{NT_NL_SPEC}, Int}()
        for (idx, spec) in enumerate(specs_small[k])
            small_lookup[spec] = idx
        end
        
        matched_count = 0
        for idx_L in 1:n_rows_large
            spec_L = specs_large[k][idx_L]
            
            if haskey(small_lookup, spec_L)
                idx_S = small_lookup[spec_L]
                W_large_new[k][idx_L, :] = W_small[k][idx_S, :]
                matched_count += 1
            end
        end
    end
    return W_large_new
end

function embed_ace_3c_params(model_small, ps_small, model_large, ps_large)

    tensor_v_small = model_small.layers.layer_1.ace_v_model.layers.ace.symbasis
    tensor_v_large = model_large.layers.layer_1.ace_v_model.layers.ace.symbasis
    W_v_small = ps_small.layer_1.acev.ace.WLL
    W_v_large_template = ps_large.layer_1.acev.ace.WLL
    
    W_v_new = transfer_component_weights(tensor_v_small, W_v_small, tensor_v_large, W_v_large_template)


    tensor_w_small = model_small.layers.layer_1.ace_w_model.layers.ace.symbasis
    tensor_w_large = model_large.layers.layer_1.ace_w_model.layers.ace.symbasis
    W_w_small = ps_small.layer_1.acew.ace.WLL
    W_w_large_template = ps_large.layer_1.acew.ace.WLL
    
    W_w_new = transfer_component_weights(tensor_w_small, W_w_small, tensor_w_large, W_w_large_template)
    
    ps_acev_new = merge(ps_large.layer_1.acev, (; ace = merge(ps_large.layer_1.acev.ace, (; WLL = tuple(W_v_new...)))))
    ps_acew_new = merge(ps_large.layer_1.acew, (; ace = merge(ps_large.layer_1.acew.ace, (; WLL = tuple(W_w_new...)))))
    

    new_layer_1 = merge(ps_large.layer_1, (; 
        acev = ps_acev_new,
        acew = ps_acew_new
    ))
    
    if hasproperty(ps_small.layer_1, :Wb)
         new_layer_1.Wb .= ps_small.layer_1.Wb
    end
    
    ps_new = merge(ps_large, (; layer_1 = new_layer_1))
    
    return ps_new
end

function embed_acev_3c_params(model_small, ps_small, model_large, ps_large)
    tensor_small = model_small.layers.layer_1.ace_model.layers.ace.symbasis
    tensor_large = model_large.layers.layer_1.ace_model.layers.ace.symbasis
    
    specs_small = get_nnll_spec(tensor_small)
    specs_large = get_nnll_spec(tensor_large)
    
    W_small = ps_small.layer_1.ace.ace.WLL
    
    W_large_new = [zeros(eltype(w), size(w)) for w in ps_large.layer_1.ace.ace.WLL]
    
    for k in 1:length(specs_large)
        n_rows_large, n_cols = size(W_large_new[k])
        small_lookup = Dict{Vector{NT_NL_SPEC}, Int}()
        for (idx, spec) in enumerate(specs_small[k])
            small_lookup[spec] = idx
        end
        
        matched_count = 0
        
        for idx_L in 1:n_rows_large
            spec_L = specs_large[k][idx_L]
            
            if haskey(small_lookup, spec_L)
                idx_S = small_lookup[spec_L]
                W_large_new[k][idx_L, :] = W_small[k][idx_S, :]
                
                matched_count += 1
            else
            end
        end
    end
    
    new_ace_inner = merge(ps_large.layer_1.ace.ace, (; WLL = tuple(W_large_new...)))
    new_ace = merge(ps_large.layer_1.ace, (; ace = new_ace_inner))
    
    new_layer_1 = merge(ps_large.layer_1, (; ace = new_ace))
    ps_new = merge(ps_large, (; layer_1 = new_layer_1))
    
    ps_new.layer_1.Wb .= ps_small.layer_1.Wb
    
    return ps_new
end