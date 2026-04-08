using Plots
using DFTK
using UnitfulAtomic
using Unitful
using EquivariantTensors
using PseudoPotentialData

function compare_bandstructure(band_data::NamedTuple, band_data2::NamedTuple;
                                unit = u"hartree")
    eshift  = something(band_data.εF, 0.0)
    data    = DFTK.data_for_plotting(band_data)
    data2   = DFTK.data_for_plotting(band_data2)
    to_unit = DFTK.ustrip(DFTK.auconvert(unit, 1.0))

    p = plot(; xlabel = "wave vector", dpi = 1000)

    # Reference bands (blue)
    first_band = true
    for σ in 1:data.n_spin, iband in 1:data.n_bands, branch in data.kbranches
        energies = (data.eigenvalues[:, iband, σ][branch] .- eshift) .* to_unit
        plot!(p, data.kdistances[branch], energies;
              color = :blue, alpha = 0.8,
              label = (first_band ? "Reference" : ""))
        first_band = false
    end

    # Predicted bands (red dashed)
    first_band2 = true
    for σ in 1:data.n_spin, iband in 1:data2.n_bands, branch in data.kbranches
        energies = (data2.eigenvalues[:, iband, σ][branch] .- eshift) .* to_unit
        plot!(p, data.kdistances[branch], energies;
              color = :red, alpha = 0.8, linestyle = :dash,
              label = (first_band2 ? "Prediction" : ""))
        first_band2 = false
    end

    # High-symmetry point vertical lines
    for branch in data.kbranches[1:end-1]
        vline!(p, [data.kdistances[last(branch)]]; color = :black, lw = 0.5, label = "")
    end

    xlims!(p, (0, data.kdistances[end]))
    xticks!(p, data.ticks.distances, data.ticks.labels)
    ylabel!(p, isnothing(band_data.εF) ?
               "eigenvalues  ($(string(unit)))" :
               "eigenvalues - εF  ($(string(unit)))")

    return p
end

function compute_and_save_bandstructures(G, sys, model, ps, st;
                                         kgrid       = [4, 4, 4],
                                         keep_bands  = 4,
                                         train_type  = "bands",
                                         psp_type    = "upf",
                                         xc_type     = "lda",
                                         temperature = 0.01,
                                         Ecut        = 15,
                                         name        = "test01")
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
    kpath = irrfbz_path(modeldft)
    bands = compute_bands(scfres, kpath; kline_density = 20u"bohr")

    # Truncate reference bands to keep_bands
    ref_bands = merge(bands, (;
        eigenvalues = [eigs[1:keep_bands] for eigs in bands.eigenvalues],
    ))

    # Build a new ETGraph with the band-path k-points
    kpoints_physical = RealMLTB.physical_k.(Ref(sys), bands.kinter)
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
    savefig(p, "bandstructure_$(name).png")
    println("Saved bandstructure_$(name).png")
    return p
end