# 3c_models.jl
#
# Factory functions that assemble complete Lux Chains:
#
#   G → [3c_layer] → (G, vH, vS)
#     → [AsmTrip2Dense] → (Hout, Sout)   # list of k-point matrices
#     → [generate_λ]    → λlist           # eigenvalues / energy bands
#
# Four variants matching the four 3c_layers:
#
#   simple_3c_model   : Wv, Wb plain parameters, Cartesian basis
#   acev_3c_model     : Wv from ACE, Wb plain parameter, spherical basis
#   ace3c_A_model     : Wv from ACE, Wb from Wpsi Case-A
#   ace3c_B_model     : Wv from ACE, Wb from Wpsi Case-B

using Random
using Lux
import LuxCore

export simple_3c_model, acev_3c_model, ace3c_A_model, ace3c_B_model

# ---------------------------------------------------------------------------
# simple_3c_model
# Cartesian basis, Wv and Wb are plain trainable parameters.
# ---------------------------------------------------------------------------
function simple_3c_model(bs_cart, NORB::Int)
    layer = Simple3Center(bs_cart, NORB)
    return Lux.Chain(
        layer,
        AsmTrip2Dense(),
        Lux.WrappedFunction(x -> generate_λ(x[1], x[2])),
    )
end

# ---------------------------------------------------------------------------
# acev_3c_model
# Spherical basis. Wv predicted by an equivariant ACE model; Wb is a parameter.
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
        Lux.WrappedFunction(x -> generate_λ(x[1], x[2])),
    )
end

# ---------------------------------------------------------------------------
# ace3c_A_model
# Spherical basis. Wv from equivariant ACE, Wb from Wpsi Case-A.
# NORB is implicit: sum_l nψ[l+1] * (2l+1).
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
        Lux.WrappedFunction(x -> generate_λ(x[1], x[2])),
    )
end

# ---------------------------------------------------------------------------
# ace3c_B_model
# Spherical basis. Wv from equivariant ACE, Wb from Wpsi Case-B.
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
        Lux.WrappedFunction(x -> generate_λ(x[1], x[2])),
    )
end