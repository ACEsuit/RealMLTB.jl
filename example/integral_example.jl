using RealMLTB
using GaussianBasis
using Molecules
using LinearAlgebra
using Random

atoms = [
    Molecules.Atom(14, 0.0, [0.0, 0.0, 0.0]),
    Molecules.Atom(14, 0.0, [2.35, 2.35, 2.35]),
]

bs = BasisSet("def2-TZVP", atoms; spherical=true, lib=:acsint)

T, V, S = compute_spherical_AO_integrals(bs)

println("size(T) = ", size(T))
println("size(V) = ", size(V))
println("size(S) = ", size(S))

nbas = size(S, 1)
Wv = randn(nbas)
Wb = Matrix(I, nbas, nbas)

H, Sout = construct_integral(Wv, Wb, T, V, S)

println("size(H) = ", size(H))
println("size(Sout) = ", size(Sout))

NORB, nat, atom_ranges = atom_ao_ranges(bs, Wb)
H12, S12 = siteblock(H, Sout, atom_ranges, 1, 2)

println("size(H12) = ", size(H12))
println("size(S12) = ", size(S12))

lmax, n_channels = lmax_and_nchannels_per_atom(bs)
println("lmax = ", lmax)
println("n_channels = ", n_channels)