using Makie
using GLMakie
using DFTK
using UnitfulAtomic
using Unitful
using EquivariantTensors
using PseudoPotentialData
using GaussianBasis

function compare_bandstructure(band_data::NamedTuple, band_data2::NamedTuple;
                                unit = u"hartree")
    eshift  = something(band_data.εF, 0.0)
    data    = DFTK.data_for_plotting(band_data)
    data2   = DFTK.data_for_plotting(band_data2)
    to_unit = DFTK.ustrip(DFTK.auconvert(unit, 1.0))

    p = Figure()
    ax = Axis(p[1,1],
              xlabel = "wave vector",
              ylabel = isnothing(band_data.εF) ?
               "eigenvalues  ($(string(unit)))" :
               "eigenvalues - εF  ($(string(unit)))",
               xticks=(data.ticks.distances, data.ticks.labels))

    # Reference bands (blue)
    first_band = true
    for σ in 1:data.n_spin, iband in 4:5, branch in data.kbranches
        energies = (data.eigenvalues[:, iband, σ][branch] .- eshift) .* to_unit
        lines!(ax, data.kdistances[branch], energies,
              color = :blue, alpha = 0.8,
              label = (first_band ? "Reference" : nothing))
        first_band = false
    end

    # Predicted bands (red dashed)
    first_band2 = true
    for σ in 1:data.n_spin, iband in 1:data2.n_bands, branch in data.kbranches
        energies = (data2.eigenvalues[:, iband, σ][branch] .- eshift) .* to_unit
        lines!(ax, data.kdistances[branch], energies,
              color = :red, alpha = 0.8, linestyle = :dash,
              label = (first_band2 ? "Prediction" : nothing))
        first_band2 = false
    end

    # High-symmetry point vertical lines
    for branch in data.kbranches[1:end-1]
        vlines!(ax, [data.kdistances[last(branch)]]; color = :black, lw = 0.5, label = "")
    end

    xlims!(ax, (0, data.kdistances[end]))
    axislegend(ax)
    return p
end

function compute_and_save_bandstructures(G, sys, model, ps, st;
                                         kgrid       = [4, 4, 4],
                                         keep_bands  = 4,
                                         train_type  = "bands",
                                         psp_type    = "upf",
                                         xc_type     = "lda",
                                         sys_type    = "graphene",
                                         temperature = 0.01,
                                         Ecut        = 15,
                                         name        = "test01",
                                         save_figure = true)
    lattice   = RealMLTB.get_lattice(sys)
    positions = RealMLTB.fractional_coordinates(sys)

    if psp_type == "upf"
        psp = PseudoFamily("dojo.nc.sr.$(xc_type).v0_4_1.standard.upf")
    elseif psp_type == "gth"
        psp = PseudoFamily("cp2k.nc.sr.$(xc_type).v0_1.semicore.gth")
    end
    element_cache = Dict{Symbol, ElementPsp}()
    atoms = map(sys.particles) do p
        sym = Symbol(species(p))
        if !haskey(element_cache, sym)
            element_cache[sym] = ElementPsp(sym, psp)
        end
        return element_cache[sym]
    end
    functional = xc_type == "lda" ? LDA() : PBE()
    modeldft = model_DFT(lattice, atoms, positions; functionals=functional, temperature)
    basis    = PlaneWaveBasis(modeldft; Ecut, kgrid)
    scfres   = self_consistent_field(basis; tol = 1e-8, callback = (_) -> nothing)

    # DFT band structure along high-symmetry path
    if sys_type == "graphene"
        kpath = irrfbz_path(modeldft; dim=2, space_group_number=13)  # graphene space group number
    else
        kpath = irrfbz_path(modeldft)
    end
    bands = compute_bands(scfres, kpath; kline_density = 20u"bohr")

    # Truncate reference bands to keep_bands
    ref_bands = merge(bands, (;
        eigenvalues = [eigs[1:keep_bands] for eigs in bands.eigenvalues],
    ))

    # Build a new ETGraph with the band-path k-points
    kinter = bands.kinter
    if sys_type == "graphene"
        kinter = map(x->SVector{3}(x..., 0), kinter)
    end
    kpoints_physical = RealMLTB.physical_k.(Ref(sys), kinter)
    G2 = EquivariantTensors.ETGraph(
        G.ii, G.jj;
        edge_data  = G.edge_data,
        graph_data = merge(G.graph_data, (; klist = kpoints_physical)),
    )

    predict = model(G2, ps, st)[1]
    if train_type == "bands"
        # ML predictions: model returns Vector{Vector{Float64}}, one per k-point
        # length = nk, each element = Vector of eigenvalues
        # Wrap predictions into a DFTK-compatible bands NamedTuple
        # eigenvalues field must be a vector of vectors (one per k-point, already sorted)
        eigenvalues = [p[1:min(keep_bands, length(p))] for p in predict[1]]
    elseif train_type == "hamiltonian"
        # ML predictions: model returns Vector{Matrix{ComplexF64}}, one per k-point
        # length = nk, each element = TB Hamiltonian matrix in k-space
        Hlist = predict.Hout
        Slist = predict.Sout
        λlist = RealMLTB.generate_λ(Hlist, Slist)
        eigenvalues = [λ[1:min(keep_bands, length(λ))] for λ in λlist]
    end
    predict_bands = merge(bands, (;eigenvalues, ))            

    p = compare_bandstructure(ref_bands, predict_bands)
    if save_figure
        save("bandstructure_$(name).png", p)
        println("Saved bandstructure_$(name).png")
    end
    return p
end

function plot_and_save_orbitals(G, model, ps, st, bs;
                                xrange = (-8, 8),
                                yrange = xrange,
                                zrange = xrange,
                                nx = 50,
                                ny = nx,
                                nz = nx,
                                iatom       = 1,
                                orb_index   = 1,
                                orb_name    = nothing,   
                                name        = "test01",
                                save_figure = true)
    @assert 1 ≤ iatom ≤ bs.natoms
    predict = model(G, ps, st)[1]
    Wb = predict.Wb
    Norb = div(size(Wb, 1), bs.natoms)

    @assert 1 ≤ orb_index ≤ Norb
    iorb = (iatom - 1) * Norb + orb_index
    iW = Wb[iorb, :]

    xs = range(xrange..., length=nx)
    ys = range(yrange..., length=ny)
    zs = range(zrange..., length=nz)
    xyzgrids = Iterators.product(xs, ys, zs)
    data3d = sum(1:bs.nbas) do j
        iW[j] * map(x->atomic_orbital_amplitude(bs, j, collect(x)), xyzgrids)
    end

    v_min = minimum(data3d)
    v_max = maximum(data3d)
    v_limit = max(abs(v_min), abs(v_max))
    custom_rwb_colors = [:red, :white, :blue]
    my_colorrange = (-v_limit, v_limit)

    p = Figure()
    orb_names = ["s", "px", "py", "pz"]
    orb_name = something(orb_name, orb_names[orb_index])
    Axis3(p[1, 1], title=orb_name)
    contour!(xrange[1]..xrange[end],
             yrange[1]..yrange[end],
             zrange[1]..zrange[end],
             data3d, levels = 6,  
             colormap = custom_rwb_colors, 
             colorrange = my_colorrange,
             alpha = 1.0,          
             transparency = true)

    if save_figure
        save("orbital_$(name).png", p)
        println("Saved orbital_$(name).png")
    end
    return p
end

# TODO: atomic_orbital_value (copied from GaussianBasis.jl), will be used directly after update
function legendre_polynomial(m::Integer,l::Integer,x)
    fmm = (-1)^m * prod((2m-1):-2:1)
    Pmm = fmm * (1 - x^2)^(m/2)
    if l == m
        return Pmm
    end
    Pmm1 = x * (2m+1) * Pmm

    if l < m || m < 0
        throw(DomainError("Associated Legendre polynomials are only defined for 0 ≤ m ≤ l, got m = $m, l = $l"))
    end

    Pmm2 = zero(Pmm)
    for i = m+2:l
        Pmm2 = ((2i-1)*x*Pmm1 - (i-1+m) * Pmm)/(i-m)
        Pmm = Pmm1
        Pmm1 = Pmm2
    end
    return Pmm1
end


function atomic_orbital_angular_part(shell::SphericalShell, n::Integer, d, d²)
    l = shell.l
    m = n - l - 1

    # for p shell, ordering is px, py, pz <=> m = 1,-1,0
    if l == 1
        m = (1, -1, 0)[n]
    end

    sqrtd² = sqrt.(d²)
    cosθ = map(selectdim(d, 1, 3), sqrtd²) do z, sqrtd²
        ifelse(iszero(z), zero(z), z / sqrtd²)
    end
    Nlm = (-1)^m * sqrt((2l + 1) / (4π * prod(l - abs(m) + 1:l + abs(m))))
    Plm = legendre_polynomial.(abs(m), l, cosθ)

    v = @. sqrtd²^l * Nlm * Plm
    if m != 0
        φ = [atan(x[2],x[1]) for x in eachslice(d, dims=tuple(2:ndims(d)...))]
        if m > 0
            @. v *= sqrt(2) * cos(m * φ)
        else
            @. v *= sqrt(2) * sin(-m * φ)
        end
    end

    return v
end


function atomic_orbital_amplitude(shell::BasisFunction, n::Integer, r::AbstractArray)
    T = promote_type(eltype(shell.coef), eltype(shell.exp), eltype(r), eltype(shell.atom.xyz))

    if size(r,1) != 3
        throw(DimensionMismatch("First dimension of r must be 3, got $(size(r,1))"))
    end

    v = zeros(T, size(r)[2:end])

    d = r .- shell.atom.xyz ./ Molecules.bohr_to_angstrom

    d² = dropdims(sum(x->x^2, d; dims=1); dims=1)

    for (c, e) in zip(shell.coef, shell.exp)
        @. v += c * exp(-e * d²)
    end

    v .*= atomic_orbital_angular_part(shell, n, d, d²)
    return v
end

function find_shell_m(B::BasisSet, idx::Integer)
    if idx <= 0
        throw(BoundsError(B, idx))
    end

    n = idx
    for shell in B.basis
        if n - (2shell.l +1) <= 0
            m = n - shell.l - 1

            # for p shell, ordering is px, py, pz <=> m = 1,-1,0
            if shell.l == 1
                m = (1, -1, 0)[n]
            end

            return shell, n
        end
        n -= 2shell.l + 1
    end

    throw(BoundsError(B, idx))
end

function atomic_orbital_amplitude(B::BasisSet, idx::Integer, r::AbstractArray)
    return atomic_orbital_amplitude(find_shell_m(B, idx)..., r)
end
function atomic_orbital_amplitude(B::BasisSet, idx::Integer, r::AbstractVector)
    return atomic_orbital_amplitude(B, idx, reshape(r,:,1))[1]
end
