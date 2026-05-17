module RealMLTB

using StaticArrays, LinearAlgebra, GenericLinearAlgebra
using Lux, LuxCore, Random
using Optimisers, Optim
using Zygote
using ChainRulesCore
using ChainRulesCore: NoTangent

include("utils/etgraph_builder.jl")
include("utils/etgraph_cut_builder.jl")

export EdgePayload,
       compute_pair_spherical_AO_integrals,
       nlist2graph,
       nlist2graph_cut
       
include("utils/EffectivePot.jl")
include("utils/spherical_integrals.jl")
include("utils/construct_integral.jl")

export EffectivePot,
       EffectivePot!,
       sparseEPI_1e3c,
       EPI_1e3c,
       EPI_1e3c!,
       generate_V_triple!,
       compute_pair_AO_integrals,
       compute_AO_integrals,
       compute_spherical_AO_integrals_sparse,
       cart2sph,
       cart2sph_generic,
       cart2sph_hardcoded,
       cart2sph_coeff,
       compute_spherical_AO_integrals,
       construct_integral,
       siteblock,
       atom_ao_ranges,
       lmax_and_nchannels_per_atom


include("ace/ace_common.jl")
include("ace/ace_wv.jl")
include("ace/ace_wpsi.jl")

export build_ace_caseA_Wpsi_model,
       build_ace_caseB_Wpsi_model,
       assemble_Wpsi_caseA,
       assemble_Wpsi_caseB

include("utils/assemble.jl")
include("model/3c_layers.jl")
include("model/3c_models.jl")
include("utils/assembly.jl")
include("train/train_common.jl")
include("train/train_single.jl")
include("train/train_multilevel.jl")

export multilevel_train_H

end
