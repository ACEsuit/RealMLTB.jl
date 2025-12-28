using LinearAlgebra
using Random
using Lux
import LuxCore
using EquivariantTensors

export Simple3Center, ACEv3Center, ACE3CenterCaseB, ACE3CenterCaseA

# ------------------------------------------------------------
# 1) Simple3Center: Wv and Wb are plain parameters
# ------------------------------------------------------------
struct Simple3Center{NORB, NBAS, NATOM, B} <: Lux.AbstractLuxLayer
    basisset::B
end

function LuxCore.initialparameters(rng::AbstractRNG,
                                   l::Simple3Center{NORB, NBAS, NATOM}) where {NORB, NBAS, NATOM}
    Wv = Lux.glorot_normal(rng, Float64, NBAS)
    Nb = div(NBAS, NATOM)
    Wb = Lux.glorot_normal(rng, Float64, NORB, Nb)
    return (Wv=Wv, Wb=Wb)
end

LuxCore.initialstates(rng::AbstractRNG, l::Simple3Center) = (basisset=l.basisset,)

function (l::Simple3Center{NORB, NBAS, NATOM})(G, ps, st) where {NORB, NBAS, NATOM}
    bs = st.basisset
    Wv = ps.Wv
    Wb = kron_I_atoms(NATOM, ps.Wb)
    vH, vS = _3center_triplet(G, bs, Wv, Wb)
    return (G, vH, vS), ps, st
end

function kron_I_atoms(NATOM::Int, Wb_atom::AbstractMatrix)
    return kron(Matrix(I, NATOM, NATOM), Wb_atom)
end

# ------------------------------------------------------------
# 2) ACEv3Center: Wv from ACE model, Wb is plain parameter
# ------------------------------------------------------------
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
    ps_ace = LuxCore.initialparameters(rng, l.ace_model) |> EquivariantTensors.float32
    Nb = div(NBAS, NATOM)
    Wb = Lux.glorot_normal(rng, Float64, NORB, Nb)
    return (ace=ps_ace, Wb=Wb)
end

function LuxCore.initialstates(rng::AbstractRNG, l::ACEv3Center)
    st_ace = LuxCore.initialstates(rng, l.ace_model) |> EquivariantTensors.float32
    return (ace=st_ace, basisset=l.basisset)
end

function (l::ACEv3Center{M,B,NORB,NBAS,NATOM})(G, ps, st) where {M,B,NORB,NBAS,NATOM}
    val_tuple, st_ace = l.ace_model(G, ps.ace, st.ace)
    Wv = assemble_Wv(val_tuple)
    Wb = kron_I_atoms(NATOM, ps.Wb)

    vH, vS = _3center_triplet(G, st.basisset, Wv, Wb)
    new_st = (ace=st_ace, basisset=st.basisset)
    return (G, vH, vS), ps, new_st
end


# ------------------------------------------------------------
# 3) ACE3CenterCaseA: Wv from ACE, Wb from Wpsi CaseA
# ------------------------------------------------------------
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
    return ACE3CenterCaseA{typeof(ace_v_model), typeof(wpsi_model), typeof(bs), NBAS, NATOM}(
        ace_v_model, wpsi_model, bs, n_channels_φ, nψ
    )
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
    val_v, st_v = l.ace_v_model(G, ps.acev, st.acev)
    val_ψ, st_ψ = l.wpsi_model(G, ps.wpsi, st.wpsi)

    Wv = assemble_Wv(val_v)
    # CaseA model output is typically (val_scalar,), so use [1]
    Wb = assemble_Wpsi_caseA(val_ψ[1], l.n_channels_φ, l.nψ)

    vH, vS = _3center_triplet(G, st.basisset, Wv, Wb)
    new_st = (acev=st_v, wpsi=st_ψ, basisset=st.basisset)
    return (G, vH, vS), ps, new_st
end

# ------------------------------------------------------------
# 4) ACE3CenterCaseB: Wv from ACE, Wb from Wpsi CaseB
# ------------------------------------------------------------
struct ACE3CenterCaseB{Mv, Mψ, B, NORB, NBAS, NATOM} <: Lux.AbstractLuxLayer
    ace_v_model::Mv
    wpsi_model::Mψ
    basisset::B
    n_channels_φ::Any
end

function ACE3CenterCaseB(ace_v_model, wpsi_model, bs, NORB::Int, n_channels_φ)
    NBAS  = bs.nbas
    NATOM = bs.natoms
    return ACE3CenterCaseB{typeof(ace_v_model), typeof(wpsi_model), typeof(bs), NORB, NBAS, NATOM}(
        ace_v_model, wpsi_model, bs, n_channels_φ
    )
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
    val_v, st_v = l.ace_v_model(G, ps.acev, st.acev)
    val_ψ, st_ψ = l.wpsi_model(G, ps.wpsi, st.wpsi)

    Wv = assemble_Wv(val_v)
    Wb = assemble_Wpsi_caseB(val_ψ, l.n_channels_φ, NORB)

    vH, vS = _3center_triplet(G, st.basisset, Wv, Wb)
    new_st = (acev=st_v, wpsi=st_ψ, basisset=st.basisset)
    return (G, vH, vS), ps, new_st
end

