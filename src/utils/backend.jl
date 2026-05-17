import EquivariantTensors as ET
import ChainRulesCore
import MLDataDevices
using ChainRulesCore: NoTangent

export CPUBackend,
       GPUBackend,
       make_backend,
       isgpu,
       graph_schema,
       enforce_backend_compatibility,
       prepare_graph,
       prepare_graphs,
       prepare_data,
       prepare_params_state,
       fix_sparsematcsx,
       to_cpu,
       gpu_cut_integral_graph

# ============================================================================
# Backend types
# ============================================================================

struct CPUBackend end

struct GPUBackend{D}
    dev::D
end

isgpu(::CPUBackend) = false
isgpu(::GPUBackend) = true

# ============================================================================
# Backend construction
# ============================================================================

function make_backend(use_cuda::Bool)
    if use_cuda
        @eval begin
            using CUDA
            import GPUArrays
        end

        CUDA.functional() || error("CUDA is requested but CUDA.functional() is false.")

        GPUArrays.allowscalar(false)

        _install_et_ad_patches()
        _install_et_cuda_patches()

        return GPUBackend(MLDataDevices.CUDADevice())
    else
        _install_et_ad_patches()
        return CPUBackend()
    end
end

# ============================================================================
# AD patch: graph geometry is fixed data, not an optimization variable
# ============================================================================

function _install_et_ad_patches()
    @eval begin
        function ChainRulesCore.rrule(
            edge::ET.EdgeEmbed,
            G::ET.ETGraph,
            ps,
            st,
        )
            y, st2 = edge(G, ps, st)

            function pb(ȳst)
                return ChainRulesCore.NoTangent(),
                       ChainRulesCore.NoTangent(),
                       ChainRulesCore.NoTangent(),
                       ChainRulesCore.NoTangent()
            end

            return (y, st2), pb
        end
    end

    return nothing
end

# ============================================================================
# CUDA patches for current EquivariantTensors sparse ACE kernels
# ============================================================================

function _install_et_cuda_patches()
    @eval begin
        # Forward: PooledSparseProduct.
        function ET._ka_evaluate_launcher!(
            A::CUDA.CuArray{TA, 2},
            basis::ET.PooledSparseProduct{NB},
            BB::Tuple{Vararg{AbstractArray{T, 3}}},
            spec::Vector{<:Tuple},
            nX::Integer,
            nneig::Integer,
        ) where {TA, NB, T}
            spec_gpu = CUDA.CuArray(spec)

            return ET._ka_evaluate_launcher!(
                A,
                basis,
                BB,
                spec_gpu,
                nX,
                nneig,
            )
        end

        # Forward: SparseSymmProd.
        function ET._ka_evaluate_launcher!(
            AA::CUDA.CuArray{TA, 2},
            basis::ET.SparseSymmProd{NB},
            A::CUDA.CuArray{TB, 2},
            specs::Tuple{Vararg{Vector{<:Tuple}}},
            nX::Integer,
        ) where {TA, TB, NB}
            specs_gpu = map(CUDA.CuArray, specs)

            return ET._ka_evaluate_launcher!(
                AA,
                basis,
                A,
                specs_gpu,
                nX,
            )
        end

        # Backward: SparseSymmProd.
        function ET.ka_pullback!(
            ∂A::CUDA.CuArray{TA, 2},
            ∂AA::CUDA.CuArray{TB, 2},
            basis::ET.SparseSymmProd{NB},
            A::CUDA.CuArray{TC, 2},
            specs::Tuple{Vararg{Vector{<:Tuple}}},
            nX::Integer,
        ) where {TA, TB, TC, NB}
            specs_gpu = map(CUDA.CuArray, specs)

            return ET.ka_pullback!(
                ∂A,
                ∂AA,
                basis,
                A,
                specs_gpu,
                nX,
            )
        end

        # Backward: PooledSparseProduct.
        function ET.ka_pullback!(
            ∂BB::Tuple,
            ∂A::CUDA.CuArray{TA, 2},
            basis::ET.PooledSparseProduct{NB},
            BB::Tuple{Vararg{AbstractArray{T, 3}}},
            spec::Vector{<:Tuple},
            nX::Integer,
            nneig::Integer,
        ) where {TA, NB, T}
            spec_gpu = CUDA.CuArray(spec)

            return ET.ka_pullback!(
                ∂BB,
                ∂A,
                basis,
                BB,
                spec_gpu,
                nX,
                nneig,
            )
        end
    end

    return nothing
end

# ============================================================================
# Graph schema detection
# ============================================================================

_has_graph_field(G, name::Symbol) =
    hasproperty(G, :graph_data) && hasproperty(G.graph_data, name)

_is_integral_table_graph(G) = _has_graph_field(G, :integral_table)
_is_cut_integral_graph(G) = _has_graph_field(G, :cut_integrals)
_is_gpu_3center_graph(G) = _has_graph_field(G, :threecenter_cache)

function graph_schema(G)
    if _is_gpu_3center_graph(G)
        return :gpu_3center
    elseif _is_integral_table_graph(G)
        return :integral_table
    elseif _is_cut_integral_graph(G)
        return :cut_integrals
    else
        return :unknown
    end
end

function _graph_schema_summary(G_list)
    counts = Dict{Symbol, Int}()

    for G in G_list
        s = graph_schema(G)
        counts[s] = get(counts, s, 0) + 1
    end

    return counts
end

# ============================================================================
# Backend compatibility
# ============================================================================

function enforce_backend_compatibility(backend::CPUBackend, G_list)
    return backend
end

function enforce_backend_compatibility(backend::GPUBackend, G_list)
    schemas = _graph_schema_summary(G_list)

    if get(schemas, :unknown, 0) > 0
        @warn """
        Some graphs have unknown graph_data schema.

        Falling back to CPUBackend.

        Graph schema counts:
            $schemas
        """
        return CPUBackend()
    end

    if get(schemas, :cut_integrals, 0) > 0
        @warn """
        This dataset uses graph_data.cut_integrals.

        Using the experimental cut_integrals GPU data path.
        If downstream assembly code contains CPU-only allocations, a later
        error may still occur and must be fixed at that assembly site.

        Graph schema counts:
            $schemas
        """
    end

    return backend
end

# ============================================================================
# ETGraph storage replacement helpers
# ============================================================================

function _backend_replace_graph_data_preserve_storage(G, new_graph_data)
    fns = fieldnames(typeof(G))

    @assert :graph_data in fns

    vals = ntuple(length(fns)) do k
        f = fns[k]
        if f === :graph_data
            new_graph_data
        else
            getfield(G, f)
        end
    end

    return (Base.typename(typeof(G)).wrapper)(vals...)
end

function _backend_replace_edge_data_preserve_storage(G, new_edge_data)
    fns = fieldnames(typeof(G))

    @assert :edge_data in fns

    vals = ntuple(length(fns)) do k
        f = fns[k]
        if f === :edge_data
            new_edge_data
        else
            getfield(G, f)
        end
    end

    return (Base.typename(typeof(G)).wrapper)(vals...)
end

# ============================================================================
# cut_integrals GPU graph
# ============================================================================

function _to_svec3_T(x, ::Type{T}) where {T}
    return typeof(x)(T.(x))
end

function _gpu_dict_arrays(dev, d::AbstractDict; T = Float32)
    out = Dict{keytype(d), Any}()

    for (k, v) in d
        if v isa AbstractArray{<:Complex}
            out[k] = dev(ComplexF32.(v))
        elseif v isa AbstractArray{<:Real}
            out[k] = dev(T.(v))
        else
            out[k] = v
        end
    end

    return out
end

function _gpu_cut_2c_entry(dev, e; T = Float32)
    return (
        integral_idx = e.integral_idx,
        Rvec         = _to_svec3_T(e.Rvec, T),
        z            = e.z,
        R_cell       = _to_svec3_T(e.R_cell, T),
        w            = T(e.w),
        V2           = dev(T.(e.V2)),
    )
end

function _gpu_cut_3c_entry(dev, e; T = Float32)
    Rij, Rik = e.Rpair

    return (
        integral_idx = e.integral_idx,
        Rpair        = (_to_svec3_T(Rij, T), _to_svec3_T(Rik, T)),
        z            = e.z,
        R_cell       = _to_svec3_T(e.R_cell, T),
        w            = T(e.w),
        V3           = dev(T.(e.V3)),
    )
end

function _gpu_cut_integrals(dev, cut; T = Float32)
    return (
        T_off = [_gpu_cut_2c_entry(dev, e; T = T) for e in cut.T_off],
        S_off = [_gpu_cut_2c_entry(dev, e; T = T) for e in cut.S_off],
        V_off = [_gpu_cut_3c_entry(dev, e; T = T) for e in cut.V_off],
    )
end

function _strip_cut_graph_data_for_gpu(G)
    return _backend_replace_graph_data_preserve_storage(G, (; klist = G.graph_data.klist))
end

"""
    gpu_cut_integral_graph(dev, G; T=Float32)

Create a GPU graph for graph_data.cut_integrals graphs.

This keeps the ACE graph payload on the device and attaches a graph_data object
whose numerical integral payloads are CuArrays.

This is not the same as gpu_3center_graph(...). It preserves the cut_integrals
schema expected by the CPU cut assembly path, but with the numerical arrays on
GPU.
"""
function gpu_cut_integral_graph(dev, G; T = Float32)
    @assert hasproperty(G.graph_data, :cut_integrals)

    # Move only ACE graph payload first.  Avoid trying to adapt the full
    # cut_integrals graph_data through EquivariantTensors.float32.
    G_ace = _strip_cut_graph_data_for_gpu(G)
    G_gpu = dev(ET.float32(G_ace))

    gd = G.graph_data

    new_graph_data = (
        klist         = G_gpu.graph_data.klist,
        T_on         = _gpu_dict_arrays(dev, gd.T_on; T = T),
        V_on         = _gpu_dict_arrays(dev, gd.V_on; T = T),
        S_on         = _gpu_dict_arrays(dev, gd.S_on; T = T),
        atom_species = gd.atom_species,
        cut_integrals = _gpu_cut_integrals(dev, gd.cut_integrals; T = T),
    )

    return _backend_replace_graph_data_preserve_storage(G_gpu, new_graph_data)
end

# ============================================================================
# Graph preparation
# ============================================================================

prepare_graph(::CPUBackend, G, NORB; T = Float64) = G

function prepare_graph(backend::GPUBackend, G, NORB; T = Float32)
    schema = graph_schema(G)

    if schema == :gpu_3center
        return G

    elseif schema == :integral_table
        return gpu_3center_graph(backend.dev, G, NORB; T = T)

    elseif schema == :cut_integrals
        return gpu_cut_integral_graph(backend.dev, G; T = T)

    else
        error("""
        Unknown graph schema.

        propertynames(G.graph_data) = $(propertynames(G.graph_data))
        """)
    end
end

function prepare_graphs(backend, G_list, NORB; T = isgpu(backend) ? Float32 : Float64)
    return [prepare_graph(backend, G, NORB; T = T) for G in G_list]
end

# ============================================================================
# Parameter/state preparation
# ============================================================================

prepare_params_state(::CPUBackend, ps, st) = ps, st

function prepare_params_state(backend::GPUBackend, ps, st)
    ps_gpu = gpu_3center_params(backend.dev, ps)
    st_gpu = gpu_3center_state(backend.dev, st)
    st_gpu = fix_sparsematcsx(backend, st_gpu)

    return ps_gpu, st_gpu
end

# ============================================================================
# Reference data preparation
# ============================================================================

prepare_data(::CPUBackend, x) = x

function prepare_data(backend::GPUBackend, x)
    return _gpu_tree(backend.dev, x)
end

function _gpu_tree(dev, x)
    if x isa AbstractArray{<:Complex}
        return dev(ComplexF32.(x))

    elseif x isa AbstractArray{<:Real}
        return dev(Float32.(x))

    elseif x isa NamedTuple
        return NamedTuple{keys(x)}(map(y -> _gpu_tree(dev, y), values(x)))

    elseif x isa Tuple
        return map(y -> _gpu_tree(dev, y), x)

    elseif x isa AbstractVector
        return map(y -> _gpu_tree(dev, y), x)

    else
        return x
    end
end

# ============================================================================
# SparseMatCSX fix
# ============================================================================

function fix_sparsematcsx(::CPUBackend, x)
    return x
end

function fix_sparsematcsx(backend::GPUBackend, x)
    if x isa ET.SparseMatCSX
        vals = map(fieldnames(typeof(x))) do f
            v = getfield(x, f)
            v isa AbstractVector ? backend.dev(v) : v
        end

        return ET.SparseMatCSX(vals...)

    elseif x isa NamedTuple
        return NamedTuple{keys(x)}(map(y -> fix_sparsematcsx(backend, y), values(x)))

    elseif x isa Tuple
        return map(y -> fix_sparsematcsx(backend, y), x)

    else
        return x
    end
end

# ============================================================================
# Move data/parameters back to CPU
# ============================================================================

function to_cpu(x)
    if isdefined(@__MODULE__, :CUDA) && x isa CUDA.CuArray
        return Array(x)

    elseif x isa NamedTuple
        return NamedTuple{keys(x)}(map(to_cpu, values(x)))

    elseif x isa Tuple
        return map(to_cpu, x)

    elseif x isa AbstractVector
        return map(to_cpu, x)

    else
        return x
    end
end
