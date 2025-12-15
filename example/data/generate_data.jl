# Data Generation 
# Computes reference eigenvalues (band structure) for training using DFTK.

using UnitfulAtomic, Unitful
using DFTK
using StaticArrays
using AtomsBase
using PseudoPotentialData
using GaussianBasis

function extract_k_and_eigenvalues(basis, scfres; keep_bands::Int = 4)
    k_list = [SVector{3,Float64}(kp.coordinate) for kp in basis.kpoints]
    λ_matrix = stack(scfres.eigenvalues)
    λ_targets = [sort(λ_matrix[1:keep_bands, i]) for i in eachindex(k_list)]
    return k_list, λ_targets
end

function extract_k_and_eigenvalues(bands; keep_bands::Int = 4)
    k_list = [SVector{3,Float64}(kp) for kp in bands.kinter]
    λ_matrix = stack(bands.eigenvalues)
    λ_targets = [sort(λ_matrix[1:keep_bands, i]) for i in eachindex(k_list)]
    return k_list, λ_targets
end

function generate_data(sys, kgrid; keep_bands::Int = 4)
    dataset = Vector{Tuple{Any, SVector{keep_bands, Float64}}}()

    lattice = RealMLTB.get_lattice(sys)
    positions = RealMLTB.fractional_coordinates(sys)
    Si = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    atoms = [Si, Si]
    model = model_DFT(lattice, atoms, positions;
                          functionals = LDA(), temperature = 0.01)
    basis = PlaneWaveBasis(model; Ecut = 15, kgrid = kgrid)
    scfres = self_consistent_field(basis, tol=1e-8, callback = (_) -> nothing)

    klist, λ_list = extract_k_and_eigenvalues(basis, scfres; keep_bands)
    for (k, λ) in zip(klist, λ_list)
        push!(dataset, (k, λ))
    end
    return dataset
end

function generate_band_data(sys, kgrid; keep_bands::Int = 4)
    dataset = Vector{Tuple{Any, SVector{keep_bands, Float64}}}()

    lattice = RealMLTB.get_lattice(sys)
    positions = RealMLTB.fractional_coordinates(sys)
    Si = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    atoms = [Si, Si]

    model = model_DFT(lattice, atoms, positions;
                          functionals = LDA(), temperature = 0.01)
    basis = PlaneWaveBasis(model; Ecut = 15, kgrid = kgrid)
    scfres = self_consistent_field(basis, tol=1e-8, callback = (_) -> nothing)

    kpath = irrfbz_path(model)
    bands = compute_bands(scfres, kpath; kline_density=20u"bohr")
    
    klist, λ_list = extract_k_and_eigenvalues(bands; keep_bands)
    for (k, λ) in zip(klist, λ_list)
        push!(dataset, (k, λ))
    end
    return dataset
end
