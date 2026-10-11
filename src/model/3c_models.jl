# 3c_models.jl
#
# Factory functions that assemble complete Lux Chains:
#
#   G → [3c_layer] → (G, vH, vS)
#     → [AsmTrip2Dense] → (Hout, Sout)
#     → [WrappedFunction] → (λ, Vlist, Hout, Sout, Wb)
#
# The final WrappedFunction returns
#   (λ = λlist, Vlist = Vlist, Hout = Hout, Sout = Sout, Wb = Wb)
# so RMSE and RMSE_bands read out.λ, and RMSE_H reads out.Wb, out.Sout and
# out.Vlist.

using Random
using Lux
import LuxCore

export simple_3c_model, acev_3c_model, ace3c_A_model, ace3c_B_model

function _final_layer(x)
    Hout, Sout, Wb = x
    λlist, Vlist = generate_λ_and_vecs(Hout, Sout)
    return (λ = λlist, Vlist = Vlist, Hout = Hout, Sout = Sout, Wb = Wb)
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
# `wb_n_out = (n_0, n_1, ...)` switches the reduction to the multiplet
# parameterisation and makes the output eigenvalues rotation invariant; NORB is
# then Σ_l n_out[l+1](2l+1) and is checked against the NORB passed in.
# `wb_n_out = nothing` keeps the dense Wb, which is NOT rotation covariant.
function acev_3c_model(bs_sph, NORB::Int, rcut;
                       Dtot::Int = 9,
                       maxl::Int = 5,
                       ORD::Int  = 3,
                       wb_n_out = nothing,
                       rng::AbstractRNG = Random.default_rng())
    _, n_channels = lmax_and_nchannels_per_atom(bs_sph)
    ace_model, _, _ = build_ace_equivariant_model(
        n_channels, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
    return Lux.Chain(
        ACEv3Center(ace_model, bs_sph, NORB; wb_n_out = wb_n_out),
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