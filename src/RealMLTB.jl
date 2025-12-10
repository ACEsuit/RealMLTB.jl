module RealMLTB

using StaticArrays, LinearAlgebra, GenericLinearAlgebra
using Lux, LuxCore, Random
using Optimisers, Optim
using Zygote
using ChainRulesCore
using ChainRulesCore: NoTangent


include("EffectivePot.jl")
include("utils.jl")
include("basic_model.jl")
include("assembly.jl")
include("train.jl")

end
