using LinearAlgebra
using StaticArrays
using ChainRulesCore
using Zygote
using Optimisers: destructure
using MatrixFuns

export RMSE, RMSE_H

# ===========================================================================
# 1) Eigenvalue-based RMSE  (used in Phase 1 + Phase 2 training)
#    Model output is now a NamedTuple: access eigenvalues via out.λ
# ===========================================================================

# --- single-sample ---
function RMSE(G, λ_ref, model, ps, st; keep_bands = 4, λ_reg = 1e-4)
    result = model(G, ps, st); out = result[1]
    λ_pred = out.λ
    if any(v -> !all(isfinite, v), λ_pred)
        return eltype(first(λ_pred))(1e6)
    end
    λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))
    sq_diff = sum(v -> sum(abs2, v), λ_pred_keep .- λ_ref)
    N       = length(λ_pred) * keep_bands
    rmse    = sqrt(sq_diff / N)
    x, _  = destructure(ps)
    reg   = λ_reg * sum(abs2, x)
    return rmse + reg
end

# --- multi-sample ---
function RMSE(Glist::AbstractVector, λ_ref_list::AbstractVector,
              model, ps, st; keep_bands = 4, λ_reg = 1e-4)
    @assert length(Glist) == length(λ_ref_list)
    total_sq = 0.0
    total_n  = 0
    for i in eachindex(Glist)
        result = model(Glist[i], ps, st); out = result[1]
        λ_pred = out.λ
        if any(v -> !all(isfinite, v), λ_pred)
            return eltype(first(λ_pred))(1e6)
        end
        λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))
        diff     = λ_pred_keep .- λ_ref_list[i]
        total_sq += sum(v -> sum(abs2, v), diff)
        total_n  += length(diff) * keep_bands
    end
    rmse  = sqrt(total_sq / total_n)
    x, _  = destructure(ps)
    reg   = λ_reg * sum(abs2, x)
    return rmse + reg
end

# ===========================================================================
# 2) Hamiltonian operator-norm RMSE  (used in Phase 3 fine-tuning)
#
#    loss = ‖f(Wb·H_ref_AO·Wb') − f(H_R0)‖_op
#
#    where:
#      H_R0      = (1/Nk)Σ_k H(k) in ψ-basis  — from model output
#      H_ref_AO  = H^proj in φ-basis            — from build_ref_and_graph_lists_*
#      Wb        = compressed-basis matrix       — re-computed via layer_1 forward
#
#    Wb is re-computed by running layer_1 again (NOT passed through the pipeline)
#    so that gradients flow correctly through ps → Wb → loss.
#
#    For ACEv3Center:    Wb = kron_I_atoms(natom, ps.layer_1.Wb)
#    For ACE3CenterCaseA/B: Wb is assembled from wpsi ACE output
#
#    We handle this generically by calling layer_1 forward and extracting Wb
#    from its intermediate computation using a small helper.
# ===========================================================================

"""
    _extract_Wb(model, G, ps, st)

Re-run layer_1 to extract the current Wb matrix.
Works for all four layer types (Simple3Center, ACEv3Center, CaseA, CaseB).
"""
function _extract_Wb(model, G, ps, st)
    layer1 = model.layers.layer_1
    ps1    = ps.layer_1
    st1    = st.layer_1

    # Run layer_1 forward — returns (G, vH, vS), ps, new_st
    # We only need Wb, which we get by calling the layer's internal logic
    return _compute_Wb(layer1, G, ps1, st1)
end

# --- Simple3Center: Wb = kron(I, ps.Wb) ---
function _compute_Wb(l::Simple3Center, G, ps, st)
    natom = EquivariantTensors.nnodes(G)
    return kron_I_atoms(natom, ps.Wb)
end

# --- ACEv3Center: Wb = kron(I, ps.Wb) ---
function _compute_Wb(l::ACEv3Center, G, ps, st)
    natom = EquivariantTensors.nnodes(G)
    return kron_I_atoms(natom, ps.Wb)
end

# --- ACE3CenterCaseA: Wb from wpsi ACE ---
function _compute_Wb(l::ACE3CenterCaseA, G, ps, st)
    val_ψ, _ = _ace_Wv_forward(l.wpsi_model, G, ps.wpsi, st.wpsi)
    val_ψ_scalar = if val_ψ isa AbstractMatrix
        val_ψ
    elseif val_ψ isa Tuple && length(val_ψ) == 1
        val_ψ[1]
    elseif val_ψ isa NamedTuple
        getfield(val_ψ, first(fieldnames(typeof(val_ψ))))
    else
        error("_compute_Wb CaseA: unexpected val_ψ type $(typeof(val_ψ))")
    end
    return assemble_Wpsi_caseA(val_ψ_scalar, l.n_channels_φ, l.nψ)
end

# --- ACE3CenterCaseB: Wb from wpsi ACE ---
function _compute_Wb(l::ACE3CenterCaseB{Mv,Mψ,B,NORB,NBAS,NATOM}, G, ps, st) where {Mv,Mψ,B,NORB,NBAS,NATOM}
    val_ψ, _ = _ace_Wv_forward(l.wpsi_model, G, ps.wpsi, st.wpsi)
    return assemble_Wpsi_caseB(val_ψ, l.n_channels_φ, NORB)
end

"""
    RMSE_H(G, H_ref_AO, model, ps, st, filter; H_reg=1e-4)

Single-sample Hamiltonian RMSE.
Computes ‖f(Wb·H_ref_AO·Wb') − f(H_R0)‖_op where Wb is re-derived from ps
so gradients flow correctly during Phase 3 training.
"""
function RMSE_H(G, H_ref_AO::AbstractMatrix,
                model, ps, st, filter::Function; H_reg=1e-4)
    # H_R0 from full model forward (gradients flow through ps → vH,vS → H_R0)
    result = model(G, ps, st); out = result[1]
    H_R0 = out.H_R0

    # Wb re-computed from layer_1 (gradients flow through ps → Wb)
    Wb = _extract_Wb(model, G, ps, st)

    H_ref_proj = Wb * real.(H_ref_AO) * Wb'
    H_ref_proj = (H_ref_proj + H_ref_proj') / 2
    H_R0_real  = real.(H_R0)
    H_R0_real  = (H_R0_real + H_R0_real') / 2
    ΔfH = mat_fun(filter, H_ref_proj) - mat_fun(filter, H_R0_real)
    return opnorm(ΔfH)
end

"""
    RMSE_H(Glist, H_ref_AO_list, model, ps, st, filter; H_reg=1e-4)

Multi-sample Hamiltonian RMSE — average over all structures.
"""
function RMSE_H(Glist::AbstractVector, H_ref_AO_list::AbstractVector,
                model, ps, st, filter::Function; H_reg=1e-4)
    @assert length(Glist) == length(H_ref_AO_list)
    total = 0.0
    for i in eachindex(Glist)
        total += RMSE_H(Glist[i], H_ref_AO_list[i], model, ps, st, filter; H_reg=H_reg)
    end
    return total / length(Glist)
end

function RMSE_H2(Glist::AbstractVector, λ_ref_list::AbstractVector,
                 ψ_ref_list::AbstractVector, proj_list::AbstractVector,
                 kw_list::AbstractVector, filter_list::AbstractVector,
                 model, ps, st; H_reg=1e-4)
    @assert length(Glist) == length(λ_ref_list)
    total = 0.0
    for i in eachindex(Glist)
        total += RMSE_H2(Glist[i], λ_ref_list[i], ψ_ref_list[i], proj_list[i], 
                         kw_list[i], filter_list[i], model, ps, st; H_reg=H_reg)
    end
    return total / length(Glist)
end


function RMSE_H2(G, λ_ref, ψ_ref, proj, kweights, 
                 filter::Function, model, ps, st; H_reg=1e-4)
    # H_k from full model forward (gradients flow through ps → vH,vS → H_R0)
    result = model(G, ps, st); out = result[1]
    H_k = out.H_k

    # Wb re-computed from layer_1 (gradients flow through ps → Wb)
    Wb = _extract_Wb(model, G, ps, st)

    Δop = sum(enumerate(kweights)) do (ik, ikw)
        Jk = proj[ik] *  Wb'
        H_ik = (H_k[ik] + H_k[ik]') / 2
        fH_k_proj = Jk * mat_fun(filter, H_ik) * Jk' 
        fH_k_ref = mat_fun_eigen(filter, λ_ref[ik], ψ_ref[ik])
        ΔfH_k = fH_k_ref - fH_k_proj

        ikw * opnorm(ΔfH_k)
    end

    return Δop
end

function mat_fun_eigen(f::Function, λ, ψ)
    dof, nbands = size(ψ)
    fH = zeros(eltype(ψ), dof, dof)
    @views for i in 1:nbands
        outersum!(fH, ψ[:, i], f(λ[i]))
    end

    return fH
end

function outersum!(result::AbstractMatrix, x::AbstractVector, a::Number)
    is, js = axes(result)
    if (is != js) || (is != axes(x, 1))
        error("mismatched array sizes")
    end
    for j in js
        for i in is
            @inbounds ci, cj = x[i], x[j]
            @inbounds result[i, j] = muladd(ci * conj(cj), a, result[i, j])
        end
    end
    result
end
