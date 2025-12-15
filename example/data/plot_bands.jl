# Compares the DFT reference band structure 
# against the ML model predictions along high-symmetry paths.

using Plots
using DFTK
using UnitfulAtomic
using EquivariantTensors
using PseudoPotentialData

function compare_bandstructure(band_data::NamedTuple, band_data2::NamedTuple;
                                unit=u"hartree")
    eshift = something(band_data.εF, 0.0)
    data = DFTK.data_for_plotting(band_data)
    data2 = DFTK.data_for_plotting(band_data2)

    to_unit = DFTK.ustrip(DFTK.auconvert(unit, 1.0))
    p = plot(; xlabel="wave vector", dpi = 1000)

    first_band = true
    for σ = 1:data.n_spin, iband = 1:data.n_bands, branch in data.kbranches
        energies = (data.eigenvalues[:, iband, σ][branch] .- eshift) .* to_unit
        plot!(p, data.kdistances[branch], energies;
              color=:blue, alpha=0.8,
              label=(first_band ? "Reference" : ""))
        first_band = false
    end

    first_band2 = true
    for σ = 1:data.n_spin, iband = 1:data2.n_bands, branch in data.kbranches
        energies = (data2.eigenvalues[:, iband, σ][branch] .- eshift) .* to_unit
        plot!(p, data.kdistances[branch], energies;
              color=:red, alpha=0.8, linestyle=:dash,
              label=(first_band2 ? "Prediction" : ""))
        first_band2 = false
    end

    for branch in data.kbranches[1:end-1]
        vline!(p, [data.kdistances[last(branch)]], color=:black, lw=0.5, label="")
    end

    xlims!(p, (0, data.kdistances[end]))
    xticks!(p, data.ticks.distances, data.ticks.labels)
    ylabel!(p, isnothing(band_data.εF) ?
            "eigenvalues  ($(string(unit)))" :
            "eigenvalues - εF  ($(string(unit)))")

    return p
end

function compute_and_save_bandstructures(sys, model, ps, st; kgrid = [4,4,4], keep_bands = 4)
    lattice = RealMLTB.get_lattice(sys)
    positions = RealMLTB.fractional_coordinates(sys)

    # Define LDA pseudopotential model for Si
    Si = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    atoms = [Si, Si]
    modeldft = model_DFT(lattice, atoms, positions;
                          functionals = LDA(), temperature = 0.01)
    basis = PlaneWaveBasis(modeldft; Ecut = 15, kgrid = kgrid)
    scfres = self_consistent_field(basis, tol=1e-8, callback = (_) -> nothing)
    # Compute band structure along high-symmetry k-path
    kpath = irrfbz_path(modeldft)
    bands = compute_bands(scfres, kpath; kline_density=20u"bohr")
    kpoints = bands.kinter

    ref_bands = (; bands..., eigenvalues = [eigs[1:keep_bands] for eigs in bands.eigenvalues])

    klist = RealMLTB.physical_k.(Ref(sys), kpoints)
    new_graph_data = merge(G.graph_data, (; klist = klist))

    G2 = EquivariantTensors.ETGraph(G.ii, G.jj; edge_data = G.edge_data, graph_data = new_graph_data)

    predict = model(G2, ps, st)[1]
    predict_bands = (; bands..., eigenvalues = [eigs[1:keep_bands] for eigs in predict])

    p = compare_bandstructure(ref_bands, predict_bands)
    savefig(p, "bandstructure.png")
end
