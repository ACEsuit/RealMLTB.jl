# Data Generation 
# Computes reference eigenvalues (band structure) for training using DFTK.

using UnitfulAtomic, Unitful
using DFTK
using StaticArrays
using AtomsBase
using PseudoPotentialData
using GaussianBasis
using Unitful: Å
using AtomsBuilder: rattle!, _set_position
using AtomsBase
using LinearAlgebra

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

function perturb(sys; ϵ = 0.1)
    # Deep copy the system to avoid modifying the original
    sys_pert = deepcopy(sys)
    new_vectors = sys_pert.cell.cell_vectors

    # Apply stretch and rotation to the cell and particle positions
    stretch = diagm(1 .+ ϵ * (2rand(3) .- 1))
    # Update cell vectors
    new_vectors = ntuple(i -> stretch * new_vectors[i], 3) 
    for i in eachindex(sys_pert.particles)
        p = sys_pert.particles[i]
        sys_pert.particles[i] = _set_position(p, stretch * p.position)
    end

    rattle!(sys_pert, ϵ)

    # Return a new FlexibleSystem with the perturbed configuration
    return AtomsBase.FlexibleSystem(
        sys_pert.particles;
        cell_vectors = new_vectors,
        periodicity = sys_pert.cell.periodicity,
        data = sys_pert.data,
    )
end

function build_ref_and_graph_lists_sph(syslist, rcut, kgrid; keep_bands=4)
    λ_ref_list = Vector{Any}(undef, length(syslist))
    G_list     = Vector{Any}(undef, length(syslist))
    bs = nothing

    for (i, sys) in enumerate(syslist)
        nlist = PairList(sys, rcut)

        # positions
        X     = position(sys, :)
        X_ang = [ustrip.(u"Å", r) for r in X]

        mol = Molecules.Molecule([
            Molecules.Atom(14, 0.0, X_ang[j]) for j in eachindex(X_ang)
        ])

        bs = BasisSet("def2-SVP", mol.atoms; spherical=true, lib=:acsint)

        dataset = generate_data(sys, kgrid; keep_bands=keep_bands)
        klist   = getfield.(dataset, 1)
        λ_ref   = getfield.(dataset, 2)

        G = nlist2graph(bs, nlist, sys, klist)

        λ_ref_list[i] = λ_ref
        G_list[i]     = G
    end

    return bs, λ_ref_list, G_list
end

function build_ref_and_graph_lists_cart(syslist, rcut, kgrid; keep_bands=4)
    λ_ref_list = Vector{Any}(undef, length(syslist))
    G_list     = Vector{Any}(undef, length(syslist))
    bs = nothing
    for (i, sys) in enumerate(syslist)
        nlist = PairList(sys, rcut)

        # positions
        X     = position(sys, :)
        X_ang = [ustrip.(u"Å", r) for r in X]

        mol = Molecules.Molecule([
            Molecules.Atom(14, 0.0, X_ang[j]) for j in eachindex(X_ang)
        ])

        # Cartesian basis: spherical=false
        bs = BasisSet("def2-SVP", mol.atoms; spherical=false, lib=:acsint)

        dataset = generate_data(sys, kgrid; keep_bands=keep_bands)
        klist   = getfield.(dataset, 1)
        λ_ref   = getfield.(dataset, 2)

        G = RealMLTB.nlist2graph(bs, nlist, sys, klist)

        λ_ref_list[i] = λ_ref
        G_list[i]     = G
    end

    return bs, λ_ref_list, G_list
end
