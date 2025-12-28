using Random
using Lux
import LuxCore

export simple_3c_model, acev_3c_model, ace3c_B_model, ace3c_A_model

"""
    simple_3c_model(bs_cart, NORB)

Cart basis, direct params Wv,Wb.
"""
function simple_3c_model(bs_cart, NORB::Int)
    NBAS  = bs_cart.nbas
    NATOM = bs_cart.natoms

    model = Chain(
    Simple3Center{NORB, NBAS, NATOM, typeof(bs_cart)}(bs_cart),
    AsmTrip2Dense(),
    WrappedFunction(x -> generate_λ(x[1], x[2])))

    return model
end


"""
    acev_3c_model(bs_sph, NORB, rcut; Dtot, maxl, ORD, rng)

Spherical basis. Wv from build_ace_equivariant_model; Wb is a parameter.
"""
function acev_3c_model(bs_sph, NORB::Int, rcut;
                       Dtot::Int=9, maxl::Int=5, ORD::Int=3,
                       rng::AbstractRNG=Random.default_rng())

    _, n_channels = lmax_and_nchannels_per_atom(bs_sph)
    ace_backend, _, _ = build_ace_equivariant_model(n_channels, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)

    return Chain(
        ACEv3Center(ace_backend, bs_sph, NORB),
        AsmTrip2Dense(),
        WrappedFunction(x -> generate_λ(x[1], x[2]))
    )
end

"""
    ace3c_A_model(bs_sph, nψ, rcut; Dtot, maxl, ORD, rng)

Spherical basis. Wv from equivariant ACE, Wb from Wpsi CaseA.
Note: NORB is implied by nψ (sum(nψ[l]*(2l+1))).
"""
function ace3c_A_model(bs_sph, nψ, rcut;
                       Dtot::Int=9, maxl::Int=5, ORD::Int=3,
                       rng::AbstractRNG=Random.default_rng())

    _, n_channels = lmax_and_nchannels_per_atom(bs_sph)

    acev_backend, _, _ = build_ace_equivariant_model(n_channels, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
    wpsi_backend, _, _ = build_ace_caseA_Wpsi_model(n_channels, nψ, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)

    return Chain(
        ACE3CenterCaseA(acev_backend, wpsi_backend, bs_sph, n_channels, nψ),
        AsmTrip2Dense(),
        WrappedFunction(x -> generate_λ(x[1], x[2]))
    )
end


"""
    ace3c_B_model(bs_sph, NORB, rcut; Dtot, maxl, ORD, rng)

Spherical basis. Wv from equivariant ACE, Wb from Wpsi CaseB.
"""
function ace3c_B_model(bs_sph, NORB::Int, rcut;
                       Dtot::Int=9, maxl::Int=5, ORD::Int=3,
                       rng::AbstractRNG=Random.default_rng())

    _, n_channels = lmax_and_nchannels_per_atom(bs_sph)

    acev_backend, _, _ = build_ace_equivariant_model(n_channels, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)
    wpsi_backend, _, _ = build_ace_caseB_Wpsi_model(n_channels, NORB, rcut; Dtot=Dtot, maxl=maxl, ORD=ORD, rng=rng)

    return Chain(
        ACE3CenterCaseB(acev_backend, wpsi_backend, bs_sph, NORB, n_channels),
        AsmTrip2Dense(),
        WrappedFunction(x -> generate_λ(x[1], x[2]))
    )
end

