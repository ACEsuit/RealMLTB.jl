using LinearAlgebra
using StaticArrays
using ChainRulesCore
using Zygote
using Optimisers: destructure
using MatrixFuns

export RMSE, RMSE_H

# ---------------------------------------------------------------------------
# Eigenvalue RMSE
# ---------------------------------------------------------------------------

function RMSE(G, λ_ref, model, ps, st; keep_bands::Int = 4)
    result = model(G, ps, st)
    out = result[1]
    λ_pred = out.λ

    if any(v -> !all(isfinite, v), λ_pred)
        return eltype(first(λ_pred))(1e6)
    end

    λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))
    sq_diff = sum(v -> sum(abs2, v), λ_pred_keep .- λ_ref)
    n_total = length(λ_pred) * keep_bands
    return sqrt(sq_diff / n_total)
end

function RMSE(Glist::AbstractVector, λ_ref_list::AbstractVector,
              model, ps, st; keep_bands::Int = 4)
    @assert length(Glist) == length(λ_ref_list)

    total_sq = 0.0
    total_n = 0

    for i in eachindex(Glist)
        result = model(Glist[i], ps, st)
        out = result[1]
        λ_pred = out.λ

        if any(v -> !all(isfinite, v), λ_pred)
            return eltype(first(λ_pred))(1e6)
        end

        λ_pred_keep = getindex.(λ_pred, Ref(1:keep_bands))
        diff = λ_pred_keep .- λ_ref_list[i]
        total_sq += sum(v -> sum(abs2, v), diff)
        total_n += length(diff) * keep_bands
    end

    return sqrt(total_sq / total_n)
end

# ---------------------------------------------------------------------------
# Extract Wb from layer_1
# ---------------------------------------------------------------------------

"""
    _extract_Wb(model, G, ps, st)

Extract the current orbital mixing / basis expansion matrix `Wb` from
`model.layers.layer_1`.

For `Simple3Center` and `ACEv3Center`, `Wb` is read directly from parameters.
For `ACE3CenterCaseA/B`, `Wb` is assembled from the `wpsi` branch.
"""
function _extract_Wb(model, G, ps, st)
    layer1 = model.layers.layer_1
    ps1 = ps.layer_1
    st1 = st.layer_1
    return _compute_Wb(layer1, G, ps1, st1)
end

function _compute_Wb(l::Simple3Center, G, ps, st)
    natom = EquivariantTensors.nnodes(G)
    return kron_I_atoms(natom, ps.Wb)
end

function _compute_Wb(l::ACEv3Center, G, ps, st)
    natom = EquivariantTensors.nnodes(G)
    return kron_I_atoms(natom, ps.Wb)
end

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

function _compute_Wb(l::ACE3CenterCaseB{Mv,Mψ,B,NORB,NBAS,NATOM}, G, ps, st) where {Mv,Mψ,B,NORB,NBAS,NATOM}
    val_ψ, _ = _ace_Wv_forward(l.wpsi_model, G, ps.wpsi, st.wpsi)
    return assemble_Wpsi_caseB(val_ψ, l.n_channels_φ, NORB)
end

# ---------------------------------------------------------------------------
# Hamiltonian loss in PW space
# ---------------------------------------------------------------------------

function RMSE_H(Glist::AbstractVector,
                 λ_ref_list::AbstractVector,
                 ψ_ref_list::AbstractVector,
                 proj_list::AbstractVector,
                 kw_list::AbstractVector,
                 filter_list::AbstractVector,
                 model, ps, st)

    n = length(Glist)
    @assert length(λ_ref_list) == n
    @assert length(ψ_ref_list) == n
    @assert length(proj_list) == n
    @assert length(kw_list) == n
    @assert length(filter_list) == n

    total = 0.0
    for i in eachindex(Glist)
        total += RMSE_H(Glist[i],
                         λ_ref_list[i],
                         ψ_ref_list[i],
                         proj_list[i],
                         kw_list[i],
                         filter_list[i],
                         model, ps, st)
    end
    return total / n
end

function RMSE_H(G,
                 λ_ref,
                 ψ_ref,
                 proj,
                 kweights,
                 filter,
                 model, ps, st)

    # H_k from full model forward (gradients flow through ps → vH,vS → H_R0)
    result = model(G, ps, st)
    out = result[1]
    H_k = out.Hout

    # Wb re-computed from layer_1 (gradients flow through ps → Wb)
    Wb = _extract_Wb(model, G, ps, st)

    Δop = sum(enumerate(kweights)) do (ik, ikw)
        Jk = proj[ik] * Wb'
        ihS = ortho_lowdin_hS(Jk)
        Jk = Jk * ihS
        H_ik = (H_k[ik] + H_k[ik]') / 2
        H_ik = ihS * H_ik * ihS
        H_ik = (H_ik + H_ik') / 2
 
        #H_pw = Jk * H_ik * Jk'
        #H_pw = (H_pw + H_pw') / 2
        #fH_k_proj = mat_fun(filter, H_pw)

        fH_k_proj = Jk * mat_fun(filter, H_ik) * Jk' 
        fH_k_ref = mat_fun_eigen(filter, λ_ref[ik], ψ_ref[ik])
        ΔfH_k = fH_k_ref - fH_k_proj

        ikw * opnorm(ΔfH_k)
    end

    return Δop
end

# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------

function ortho_lowdin_hS(φk::AbstractArray{T}) where {T}
    evals, evecs = eigen(Hermitian(φk' * φk))
    @assert minimum(abs, evals) > eps(real(T)) * maximum(abs, evals)
    return evecs * Diagonal(evals .^ (-0.5)) * evecs'
end

function mat_fun_eigen(f, λ, ψ)
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
    return result
end

