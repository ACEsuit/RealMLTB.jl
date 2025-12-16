using RealMLTB
using Molecules, GaussianBasis
using AtomsBuilder, NeighbourLists
using EquivariantTensors
using Unitful: ustrip, Å
using LuxCore
using Random

include("data/generate_data.jl")
include("data/plot_bands.jl")

rcut = 12.5u"Å"
sys  = bulk(:Si)

syslist = [sys; [perturb(sys) for _ in 1:3]]
NORB  = 4
kgrid = [3, 4, 5]
keep_bands    = 4

bs = nothing
λ_ref_list = Vector{Any}(undef, length(syslist))
G_list     = Vector{Any}(undef, length(syslist))

for (i, sys) in enumerate(syslist)
    nlist = PairList(sys, rcut)

    # positions
    X     = position(sys, :)
    X_ang = [ustrip.(u"Å", r) for r in X]

    mol = Molecules.Molecule([
        Molecules.Atom(14, 0.0, X_ang[j]) for j in eachindex(X_ang)
    ])

    bs  = BasisSet("def2-SVP", mol.atoms; spherical=true, lib=:acsint)

    dataset = generate_band_data(sys, kgrid; keep_bands=keep_bands)
    klist   = getfield.(dataset, 1)
    λ_ref   = getfield.(dataset, 2)

    G = nlist2graph(bs, nlist, sys, klist)

    λ_ref_list[i] = λ_ref
    G_list[i]     = G
end

model = acev_3c_model(bs, NORB, rcut; Dtot = 5, maxl = 5, ORD = 2)
ps, st = LuxCore.setup(MersenneTwister(1234), model)

ps, x_history = train(G_list, model, ps, st, keep_bands, λ_ref_list; iterations = 3000)
error         = RMSE(G_list, λ_ref_list, model, ps, st; keep_bands = keep_bands)

compute_and_save_bandstructures(sys, model, ps, st; kgrid = kgrid, keep_bands = keep_bands)
