using RealMLTB
using GaussianBasis
using Molecules
using AtomsBuilder
using NeighbourLists
using EquivariantTensors
using Unitful
using Unitful: Å
using UnitfulAtomic: ustrip

# 1. build system
sys = bulk(:Si)

# 2. neighbour list
rcut = 4.0u"Å"
nlist = PairList(sys, rcut)

# 3. basis on one local environment
X = position(sys, :)
X_ang = [ustrip.(Å, r) for r in X]

atoms = [
    Molecules.Atom(14, 0.0, X_ang[1]),
    Molecules.Atom(14, 0.0, X_ang[2]),
]

bs = BasisSet("def2-TZVP", atoms; spherical=true, lib=:acsint)

# 4. k-points
using StaticArrays

klist = [
    SVector(0.0, 0.0, 0.0),
]

# 5. build ETGraph
G = nlist2graph(bs, nlist, sys, klist)

println("nnodes = ", EquivariantTensors.nnodes(G))
println("nedges = ", EquivariantTensors.nedges(G))
println("type(edge_data[1]) = ", typeof(G.edge_data[1]))

edata = G.edge_data[1]
println("size(edata.T_pair) = ", size(edata.T_pair))
println("size(edata.S_pair) = ", size(edata.S_pair))
println("size(edata.V_pair) = ", size(edata.V_pair))

println("size(G.graph_data.T_on) = ", size(G.graph_data.T_on))
println("size(G.graph_data.S_on) = ", size(G.graph_data.S_on))
println("size(G.graph_data.V_on) = ", size(G.graph_data.V_on))