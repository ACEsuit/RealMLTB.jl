using RealMLTB
using GaussianBasis
using Molecules
using AtomsBuilder
using NeighbourLists
using EquivariantTensors
using Unitful
using Unitful: Å
using UnitfulAtomic: ustrip
using Random
using Lux
import LuxCore
using StaticArrays
using Zygote
using LinearAlgebra

import EquivariantTensors as ET 
import EquivariantTensors: PState, VState

# using CUDA
# dev = CUDADevice()

using Metal 
dev = mtl

import GPUArrays
GPUArrays.allowscalar(false)

##
# ---------------------------------------------------------------------------
# 1. Build a system (bulk silicon)
# ---------------------------------------------------------------------------
sys = bulk(:Si)


# Define cutoff radius for neighbor list construction
rcut = 4.0u"Å"

# Build neighbor list (pair interactions within cutoff)
nlist = PairList(sys, rcut)

##
# ---------------------------------------------------------------------------
# Construct input graph 
# ---------------------------------------------------------------------------

X = position(sys, :)
X_ang = [ustrip.(Å, r) for r in X]

# Build atomic objects for GaussianBasis
atoms = [
    Molecules.Atom(14, 0.0, X_ang[1]),   # Si atom 1
    Molecules.Atom(14, 0.0, X_ang[2]),   # Si atom 2
]

# Define k-point list
klist = [
    SVector(0.0, 0.0, 0.0),
]

bs = BasisSet("STO-3g", atoms; spherical=true, lib=:acsint)

# Build graph
G = nlist2graph(bs, nlist, sys, klist)

# remove basis data from the graph 
e1 = [ PState(R_cell = e.R_cell, 𝐫 = e.𝐫, z0 = e.z0, z1 = e.z1) 
       for e in G.edge_data ]
G1 = ET.ETGraph(G.ii, G.jj, G.first, G.node_data, e1, G.graph_data, G.maxneigs)

G_gpu = dev(ET.float32(G1))

##

_, n_channels = RealMLTB.lmax_and_nchannels_per_atom(bs)
model, ps, st = RealMLTB.build_ace_equivariant_model(n_channels, rcut)

ps_gpu = dev(ET.float32(ps))
st_gpu = dev(ET.float32(st))

out, _ = model(G1, ps, st)
out_gpu, _ = model(G_gpu, ps_gpu, st_gpu)

@show max(norm.(Array.(out_gpu) .- out)...)

##

# changing the pullback test to a setting I'm a bit more 
# familiar with - compute a scalar, then take the gradient. For some reasons 
# this made it easier to read the stack and find the bug. 

function myscalar(model, G, ps, st) 
    out, _ = model(G, ps, st)
    return sum(out[1].^2) + sum(abs2, sum(out[2]))
end

val = myscalar(model, G1, ps, st)
val_gpu = myscalar(model, G_gpu, ps_gpu, st_gpu)
@show val ≈ val_gpu

g = Zygote.gradient(_ps -> myscalar(model, G1, _ps, st), ps)[1]
g_gpu = Zygote.gradient(_ps -> myscalar(model, G_gpu, _ps, st_gpu), ps_gpu)[1]


@show g.ace.WLL[1] ≈ Array(g_gpu.ace.WLL[1])
@show norm(g.ace.WLL[2] - Array(g_gpu.ace.WLL[2]))

# remaining issue to understand: why is g.WLL[2] ≈ 0 ??? 
