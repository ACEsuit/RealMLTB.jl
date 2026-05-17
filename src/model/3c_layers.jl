# 3c_layers.jl
#
# Four Lux layers that map  G → (G, vH, vS):
#
#   Simple3Center    : Wv, Wb are plain trainable parameters
#   ACEv3Center      : Wv from equivariant ACE, Wb is a plain parameter
#   ACE3CenterCaseA  : Wv from ACE, Wb from Wpsi Case-A (co-rotating)
#   ACE3CenterCaseB  : Wv from ACE, Wb from Wpsi Case-B (isotropic)
#
# All four layers support prediction on systems with a different number of
# atoms than the training system, provided the basisset in `st` is updated
# to match the new system:
#
#   st_pred = update_basisset(st, bs_new)
#   λ, _    = model(G_pred, ps, st_pred)

using LinearAlgebra
using Random
using Lux
import LuxCore
using EquivariantTensors
using GaussianBasis: BasisSet

export Simple3Center, ACEv3Center, ACE3CenterCaseA, ACE3CenterCaseB
export update_basisset, gpu_3center_params, gpu_3center_state

# ---------------------------------------------------------------------------
# Helper: broadcast a per-atom Wb_atom (NORB × Nb) to the full system
# by Kronecker product with identity over atoms.
# natom is read dynamically from G, not from a type parameter.
# ---------------------------------------------------------------------------
function kron_I_atoms(NATOM::Int, Wb_atom::AbstractMatrix)
    return kron(Matrix{eltype(Wb_atom)}(I, NATOM, NATOM), Wb_atom)
end

function _kron_I_atoms_kernel!(Wb, Wb_atom, NATOM::Int, NORB::Int, Nb::Int)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = NATOM * NORB * Nb

    if q <= N
        t = q - 1

        b = (t % Nb) + 1
        t = t ÷ Nb

        a = (t % NORB) + 1
        I = (t ÷ NORB) + 1

        row = (I - 1) * NORB + a
        col = (I - 1) * Nb   + b

        Wb[row, col] = Wb_atom[a, b]
    end

    return nothing
end

# GPU method: same public name
function kron_I_atoms(NATOM::Int, Wb_atom::CUDA.CuArray{T,2}) where {T}
    NORB, Nb = size(Wb_atom)

    Wb = CUDA.zeros(T, NATOM * NORB, NATOM * Nb)

    threads = 256
    N = NATOM * NORB * Nb
    blocks = cld(N, threads)

    @cuda threads=threads blocks=blocks _kron_I_atoms_kernel!(
        Wb, Wb_atom, NATOM, NORB, Nb
    )

    return Wb
end

function _kron_I_atoms_pullback_kernel!(ΔWb_atom, ΔWb, NATOM::Int, NORB::Int, Nb::Int)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = NORB * Nb

    if q <= N
        t = q - 1

        b = (t % Nb) + 1
        a = (t ÷ Nb) + 1

        acc = zero(eltype(ΔWb_atom))

        for I in 1:NATOM
            row = (I - 1) * NORB + a
            col = (I - 1) * Nb   + b

            acc += eltype(ΔWb_atom)(real(ΔWb[row, col]))
        end

        ΔWb_atom[a, b] = acc
    end

    return nothing
end

function ChainRulesCore.rrule(
    ::typeof(kron_I_atoms),
    NATOM::Int,
    Wb_atom::CUDA.CuArray{T,2},
) where {T}

    Wb = kron_I_atoms(NATOM, Wb_atom)

    function pullback_kron_I_atoms(ΔWb_raw)
        ΔWb = ChainRulesCore.unthunk(ΔWb_raw)

        NORB, Nb = size(Wb_atom)
        ΔWb_atom = CUDA.zeros(T, NORB, Nb)

        threads = 256
        N = NORB * Nb
        blocks = cld(N, threads)

        @cuda threads=threads blocks=blocks _kron_I_atoms_pullback_kernel!(
            ΔWb_atom, ΔWb, NATOM, NORB, Nb
        )

        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ΔWb_atom
    end

    return Wb, pullback_kron_I_atoms
end


_to_gpu_float32_preserve_basis(dev, x::Nothing) = nothing
_to_gpu_float32_preserve_basis(dev, x::BasisSet) = x
_to_gpu_float32_preserve_basis(dev, x::Bool) = x
_to_gpu_float32_preserve_basis(dev, x::Integer) = x
_to_gpu_float32_preserve_basis(dev, x::AbstractFloat) = Float32(x)
_to_gpu_float32_preserve_basis(dev, x::Complex) = ComplexF32(x)
_to_gpu_float32_preserve_basis(dev, x::Symbol) = x
_to_gpu_float32_preserve_basis(dev, x::AbstractString) = x

# Keep StaticArrays as StaticArrays. They are small structural constants/states,
# and converting them to CuArray often breaks code expecting SMatrix/SVector.
function _to_gpu_float32_preserve_basis(dev, x::StaticArrays.StaticArray)
    if eltype(x) <: Complex
        return ComplexF32.(x)
    elseif eltype(x) <: AbstractFloat
        return Float32.(x)
    else
        return x
    end
end

# Ordinary numeric arrays should live on the GPU.
function _to_gpu_float32_preserve_basis(dev, x::AbstractArray{<:Number})
    if eltype(x) <: Complex
        return dev(ComplexF32.(x))
    elseif eltype(x) <: AbstractFloat
        return dev(Float32.(x))
    else
        return dev(x)
    end
end

function _to_gpu_float32_preserve_basis(dev, x::Tuple)
    return map(v -> _to_gpu_float32_preserve_basis(dev, v), x)
end

function _to_gpu_float32_preserve_basis(dev, x::NamedTuple)
    return NamedTuple{keys(x)}(
        map(v -> _to_gpu_float32_preserve_basis(dev, v), values(x))
    )
end

# Fallback: try EquivariantTensors.float32 first; if unsupported, keep x.
function _to_gpu_float32_preserve_basis(dev, x)
    try
        return dev(EquivariantTensors.float32(x))
    catch err
        if err isa MethodError
            return x
        else
            rethrow(err)
        end
    end
end

function gpu_3center_params(dev, ps)
    return _to_gpu_float32_preserve_basis(dev, ps)
end

function gpu_3center_state(dev, st)
    return _to_gpu_float32_preserve_basis(dev, st)
end
# ---------------------------------------------------------------------------
# Helper: update the basisset stored in `st` so the model can be applied
# to a system with a different number of atoms / cell geometry.
#
# Usage:
#   st_pred = update_basisset(st, bs_new)
#   λ, _    = model(G_pred, ps, st_pred)
# ---------------------------------------------------------------------------
function update_basisset(st, bs_new)
    # The basisset lives at st.layer_1.basisset for all four layer types.
    l1_new = merge(st.layer_1, (; basisset = bs_new))
    return merge(st, (; layer_1 = l1_new))
end

# ---------------------------------------------------------------------------
# 1) Simple3Center
#    Wv ∈ ℝ^NBAS and Wb_atom ∈ ℝ^{NORB × Nb} are plain parameters.
# ---------------------------------------------------------------------------
struct Simple3Center{NORB, NBAS, NATOM, B} <: Lux.AbstractLuxLayer
    basisset::B
end

function Simple3Center(bs, NORB::Int)
    NBAS  = bs.nbas
    NATOM = bs.natoms
    return Simple3Center{NORB, NBAS, NATOM, typeof(bs)}(bs)
end

function LuxCore.initialparameters(rng::AbstractRNG,
                                   l::Simple3Center{NORB,NBAS,NATOM}) where {NORB,NBAS,NATOM}
    Nb      = div(NBAS, NATOM)
    Wv      = Lux.glorot_normal(rng, Float64, NBAS)
    Wb_atom = Lux.glorot_normal(rng, Float64, NORB, Nb)
    return (Wv=Wv, Wb=Wb_atom)
end

LuxCore.initialstates(::AbstractRNG, l::Simple3Center) = (basisset=l.basisset,)

function (l::Simple3Center{NORB,NBAS,NATOM})(G, ps, st) where {NORB,NBAS,NATOM}
    natom_actual = EquivariantTensors.nnodes(G)
    Wb   = kron_I_atoms(natom_actual, ps.Wb)
    vH, vS = _3center_triplet(G, st.basisset, ps.Wv, Wb)
    return (G, vH, vS, Wb), ps, st
end

# ---------------------------------------------------------------------------
# 2) ACEv3Center
#    Wv from equivariant ACE model; Wb_atom is a plain parameter.
# ---------------------------------------------------------------------------
struct ACEv3Center{M, B, NORB, NBAS, NATOM} <: Lux.AbstractLuxLayer
    ace_model::M
    basisset::B
end

function ACEv3Center(ace_model, bs, NORB::Int)
    NBAS  = bs.nbas
    NATOM = bs.natoms
    return ACEv3Center{typeof(ace_model), typeof(bs), NORB, NBAS, NATOM}(ace_model, bs)
end

function LuxCore.initialparameters(rng::AbstractRNG,
                                   l::ACEv3Center{M,B,NORB,NBAS,NATOM}) where {M,B,NORB,NBAS,NATOM}
    ps_ace  = LuxCore.initialparameters(rng, l.ace_model) |> EquivariantTensors.float32
    Nb      = div(NBAS, NATOM)
    Wb_atom = Lux.glorot_normal(rng, Float64, NORB, Nb)
    return (ace=ps_ace, Wb=Wb_atom)
end

function LuxCore.initialstates(rng::AbstractRNG, l::ACEv3Center)
    st_ace = LuxCore.initialstates(rng, l.ace_model) |> EquivariantTensors.float32
    return (ace=st_ace, basisset=l.basisset)
end

function (l::ACEv3Center{M,B,NORB,NBAS,NATOM})(G, ps, st) where {M,B,NORB,NBAS,NATOM}
    val_tuple, st_ace = _ace_Wv_forward(l.ace_model, G, ps.ace, st.ace)
    Wv = assemble_Wv(val_tuple)
    # use actual atom count from G, not the fixed NATOM type parameter
    natom_actual = EquivariantTensors.nnodes(G)
    Wb = kron_I_atoms(natom_actual, ps.Wb)
    vH, vS = _3center_triplet(G, st.basisset, Wv, Wb)
    new_st = (ace=st_ace, basisset=st.basisset)
    return (G, vH, vS, Wb), ps, new_st
end

# ---------------------------------------------------------------------------
# 3) ACE3CenterCaseA
#    Wv from equivariant ACE; Wb from Wpsi Case-A (co-rotating compressed basis).
# ---------------------------------------------------------------------------
struct ACE3CenterCaseA{Mv, Mψ, B, NBAS, NATOM} <: Lux.AbstractLuxLayer
    ace_v_model::Mv
    wpsi_model::Mψ
    basisset::B
    n_channels_φ::Any
    nψ::Any
end

function ACE3CenterCaseA(ace_v_model, wpsi_model, bs, n_channels_φ, nψ)
    NBAS  = bs.nbas
    NATOM = bs.natoms
    return ACE3CenterCaseA{
        typeof(ace_v_model), typeof(wpsi_model), typeof(bs), NBAS, NATOM}(
        ace_v_model, wpsi_model, bs, n_channels_φ, nψ)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::ACE3CenterCaseA)
    ps_v = LuxCore.initialparameters(rng, l.ace_v_model) |> EquivariantTensors.float32
    ps_ψ = LuxCore.initialparameters(rng, l.wpsi_model)  |> EquivariantTensors.float32
    return (acev=ps_v, wpsi=ps_ψ)
end

function LuxCore.initialstates(rng::AbstractRNG, l::ACE3CenterCaseA)
    st_v = LuxCore.initialstates(rng, l.ace_v_model) |> EquivariantTensors.float32
    st_ψ = LuxCore.initialstates(rng, l.wpsi_model)  |> EquivariantTensors.float32
    return (acev=st_v, wpsi=st_ψ, basisset=l.basisset)
end

function (l::ACE3CenterCaseA)(G, ps, st)
    val_v, st_v = _ace_Wv_forward(l.ace_v_model, G, ps.acev, st.acev)
    val_ψ, st_ψ = _ace_Wv_forward(l.wpsi_model,  G, ps.wpsi, st.wpsi)

    Wv = assemble_Wv(val_v)

    val_ψ_scalar = if val_ψ isa AbstractMatrix
        val_ψ
    elseif val_ψ isa Tuple && length(val_ψ) == 1
        val_ψ[1]
    elseif val_ψ isa NamedTuple
        getfield(val_ψ, first(fieldnames(typeof(val_ψ))))
    else
        error("ACE3CenterCaseA: unexpected val_ψ type $(typeof(val_ψ))")
    end

    Wb = assemble_Wpsi_caseA(val_ψ_scalar, l.n_channels_φ, l.nψ)

    vH, vS = _3center_triplet(G, st.basisset, Wv, Wb)
    new_st = (acev=st_v, wpsi=st_ψ, basisset=st.basisset)
    return (G, vH, vS, Wb), ps, new_st
end

# ---------------------------------------------------------------------------
# 4) ACE3CenterCaseB
#    Wv from equivariant ACE; Wb from Wpsi Case-B (isotropic scalar orbitals).
# ---------------------------------------------------------------------------
struct ACE3CenterCaseB{Mv, Mψ, B, NORB, NBAS, NATOM} <: Lux.AbstractLuxLayer
    ace_v_model::Mv
    wpsi_model::Mψ
    basisset::B
    n_channels_φ::Any
end

function ACE3CenterCaseB(ace_v_model, wpsi_model, bs, NORB::Int, n_channels_φ)
    NBAS  = bs.nbas
    NATOM = bs.natoms
    return ACE3CenterCaseB{
        typeof(ace_v_model), typeof(wpsi_model), typeof(bs), NORB, NBAS, NATOM}(
        ace_v_model, wpsi_model, bs, n_channels_φ)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::ACE3CenterCaseB)
    ps_v = LuxCore.initialparameters(rng, l.ace_v_model) |> EquivariantTensors.float32
    ps_ψ = LuxCore.initialparameters(rng, l.wpsi_model)  |> EquivariantTensors.float32
    return (acev=ps_v, wpsi=ps_ψ)
end

function LuxCore.initialstates(rng::AbstractRNG, l::ACE3CenterCaseB)
    st_v = LuxCore.initialstates(rng, l.ace_v_model) |> EquivariantTensors.float32
    st_ψ = LuxCore.initialstates(rng, l.wpsi_model)  |> EquivariantTensors.float32
    return (acev=st_v, wpsi=st_ψ, basisset=l.basisset)
end

function (l::ACE3CenterCaseB{Mv,Mψ,B,NORB,NBAS,NATOM})(G, ps, st) where {Mv,Mψ,B,NORB,NBAS,NATOM}
    val_v, st_v = _ace_Wv_forward(l.ace_v_model, G, ps.acev, st.acev)
    val_ψ, st_ψ = _ace_Wv_forward(l.wpsi_model,  G, ps.wpsi, st.wpsi)

    Wv = assemble_Wv(val_v)
    Wb = assemble_Wpsi_caseB(val_ψ, l.n_channels_φ, NORB)

    vH, vS = _3center_triplet(G, st.basisset, Wv, Wb)
    new_st = (acev=st_v, wpsi=st_ψ, basisset=st.basisset)
    return (G, vH, vS, Wb), ps, new_st
end