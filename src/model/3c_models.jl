# 3c_models.jl
#
# Factory functions that assemble complete Lux Chains:
#
#   G → [3c_layer] → (G, vH, vS)
#     → [AsmTrip2Dense] → (Hout, Sout)
#     → [WrappedFunction] → (λ = λlist, H_R0 = H_R0)
#
# CHANGE vs original: the final WrappedFunction now returns a NamedTuple
#   (λ = λlist, H_R0 = H_R0)
# so that RMSE can access λ via out.λ, and RMSE_H can access H_R0 via out.H_R0.
#
# H_R0 = (1/Nk) Σ_k H(k)  — the R=0 Hamiltonian in the ψ-basis.

using Random
using Lux
import LuxCore

export simple_3c_model, acev_3c_model, ace3c_A_model, ace3c_B_model

function _final_layer(x)
    Hout, Sout, Wb = x  
    λlist = generate_λ(Hout, Sout)
    return (λ = λlist, Hout = Hout, Sout = Sout, Wb = Wb)
end

# ---------------------------------------------------------------------------
# simple_3c_model
# ---------------------------------------------------------------------------
function simple_3c_model(bs_cart, NORB::Int)
    layer = Simple3Center(bs_cart, NORB)
    return Lux.Chain(
        layer,
        AsmTrip2Dense(),
        Lux.WrappedFunction(_final_layer),
    )
end

# ---------------------------------------------------------------------------
# acev_3c_model
# ---------------------------------------------------------------------------
function acev_3c_model(bs_sph, NORB::Int, rcut;
                       Dtot::Int = 9,
                       maxl::Int = 5,
                       ORD::Int  = 3,
                       rng::AbstractRNG = Random.default_rng())
    _, n_channels = lmax_and_nchannels_per_atom(bs_sph)
    ace_model, _, _ = build_ace_equivariant_model(
        n_channels, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
    return Lux.Chain(
        ACEv3Center(ace_model, bs_sph, NORB),
        AsmTrip2Dense(),
        Lux.WrappedFunction(_final_layer),
    )
end

# ---------------------------------------------------------------------------
# ace3c_A_model
# ---------------------------------------------------------------------------
function ace3c_A_model(bs_sph, nψ, rcut;
                       Dtot::Int = 9,
                       maxl::Int = 5,
                       ORD::Int  = 3,
                       rng::AbstractRNG = Random.default_rng())
    _, n_channels = lmax_and_nchannels_per_atom(bs_sph)
    acev_model, _, _ = build_ace_equivariant_model(
        n_channels, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
    wpsi_model, _, _ = build_ace_caseA_Wpsi_model(
        n_channels, nψ, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
    return Lux.Chain(
        ACE3CenterCaseA(acev_model, wpsi_model, bs_sph, n_channels, nψ),
        AsmTrip2Dense(),
        Lux.WrappedFunction(_final_layer),
    )
end

# ---------------------------------------------------------------------------
# ace3c_B_model
# ---------------------------------------------------------------------------
function ace3c_B_model(bs_sph, NORB::Int, rcut;
                       Dtot::Int = 9,
                       maxl::Int = 5,
                       ORD::Int  = 3,
                       rng::AbstractRNG = Random.default_rng())
    _, n_channels = lmax_and_nchannels_per_atom(bs_sph)
    acev_model, _, _ = build_ace_equivariant_model(
        n_channels, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
    wpsi_model, _, _ = build_ace_caseB_Wpsi_model(
        n_channels, NORB, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
    return Lux.Chain(
        ACE3CenterCaseB(acev_model, wpsi_model, bs_sph, NORB, n_channels),
        AsmTrip2Dense(),
        Lux.WrappedFunction(_final_layer),
    )
end