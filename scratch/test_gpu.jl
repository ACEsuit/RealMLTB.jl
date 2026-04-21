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

import EquivariantTensors as ET 

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
# 2. Construct atomic basis
# ---------------------------------------------------------------------------

X = position(sys, :)
X_ang = [ustrip.(Å, r) for r in X]

# Build atomic objects for GaussianBasis
atoms = [
    Molecules.Atom(14, 0.0, X_ang[1]),   # Si atom 1
    Molecules.Atom(14, 0.0, X_ang[2]),   # Si atom 2
]

# Construct Gaussian atomic orbital basis set
# bs = BasisSet("def2-TZVP", atoms; spherical=true, lib=:acsint)
bs = BasisSet("STO-3g", atoms; spherical=true, lib=:acsint)

##
# ---------------------------------------------------------------------------
# 3. ACE-based 3-center model
# ---------------------------------------------------------------------------

rng = MersenneTwister(1234)

# Number of target orbitals per atom (compressed basis size)
NORB = 4

# Build equivariant ACE-based 3-center model
# - Dtot: radial polynomial order
# - maxl: maximum angular momentum
# - ORD : many-body expansion order
model = acev_3c_model(bs, NORB, rcut;
    Dtot = 5,
    maxl = 5,
    ORD  = 3,
    rng = rng,
)

ps, st = LuxCore.setup(rng, model)

##
# ---------------------------------------------------------------------------
# 4. Build graph input G
# ---------------------------------------------------------------------------

# Define k-point list
klist = [
    SVector(0.0, 0.0, 0.0),
]

# Build graph
G = nlist2graph(bs, nlist, sys, klist)

##

# @profview let bs = bs, nlist = nlist, sys = sys, klist = klist 
#    for _ = 1:20 
#       G = nlist2graph(bs, nlist, sys, klist)
#    end
# end

##
# ---------------------------------------------------------------------------
# 5. Forward pass: evaluate model
# ---------------------------------------------------------------------------

# Run model to obtain:
# - λ    : eigenvalues (band energies)
# - H : Hamiltonian matrices (k-dependent)
# - S : overlap matrices
# - Wb   : AO → compressed orbital transformation matrix
out, st = model(G, ps, st)

##

e1 = [ (R_cell = e.R_cell, 𝐫 = e.𝐫, z0 = e.z0, z1 = e.z1) 
       for e in G.edge_data ]
G1 = ET.ETGraph(G.ii, G.jj, G.first, G.node_data, e1, G.graph_data, G.maxneigs)

ps_gpu = dev(ps)
st_gpu = dev(st)
G_gpu = dev(ET.float32(G1))

# out, st = model(G_gpu, ps_gpu, st_gpu)

_, n_channels = RealMLTB.lmax_and_nchannels_per_atom(bs)
model_ace, ps_ace, st_ace = RealMLTB.build_ace_equivariant_model(n_channels, rcut)

ps_ace_gpu = dev(ps_ace)
st_ace_gpu = dev(st_ace)

##

out_ace, _ = model_ace(G_gpu, ps_ace_gpu, st_ace_gpu)
