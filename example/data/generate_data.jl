using UnitfulAtomic, Unitful
using DFTK
using StaticArrays
using AtomsBase
using PseudoPotentialData
using GaussianBasis
using Unitful: Å
using AtomsBuilder: rattle!, _set_position
using LinearAlgebra

# ---------------------------------------------------------------------------
# Extract k-points and eigenvalues from a completed SCF result
# ---------------------------------------------------------------------------
function extract_k_and_eigenvalues(basis, scfres; keep_bands::Int = 4)
    k_list    = [SVector{3,Float64}(kp.coordinate) for kp in basis.kpoints]
    λ_matrix  = stack(scfres.eigenvalues)       # (nbands × nk)
    λ_targets = [Vector{Float64}(sort(λ_matrix[1:keep_bands, i]))
                 for i in eachindex(k_list)]
    return k_list, λ_targets
end

function extract_k_and_eigenvalues(bands; keep_bands::Int = 4)
    k_list    = [SVector{3,Float64}(kp) for kp in bands.kinter]
    λ_matrix  = stack(bands.eigenvalues)
    λ_targets = [Vector{Float64}(sort(λ_matrix[1:keep_bands, i]))
                 for i in eachindex(k_list)]
    return k_list, λ_targets
end

# ---------------------------------------------------------------------------
# Run DFT on sys and return (k_list, λ_targets) for a Monkhorst-Pack grid
# ---------------------------------------------------------------------------
function generate_data(sys, kgrid; keep_bands::Int = 4)
    lattice   = RealMLTB.get_lattice(sys)
    positions = RealMLTB.fractional_coordinates(sys)
    Si        = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    atoms     = fill(Si, length(positions))

    model  = model_DFT(lattice, atoms, positions;
                       functionals = LDA(), temperature = 0.01)
    basis  = PlaneWaveBasis(model; Ecut = 15, kgrid = kgrid)
    scfres = self_consistent_field(basis; tol = 1e-8, callback = (_) -> nothing)

    return extract_k_and_eigenvalues(basis, scfres; keep_bands)
end

# ---------------------------------------------------------------------------
# Run DFT and compute band structure along the irreducible BZ path
# ---------------------------------------------------------------------------
function generate_band_data(sys, kgrid; keep_bands::Int = 4)
    lattice   = RealMLTB.get_lattice(sys)
    positions = RealMLTB.fractional_coordinates(sys)
    Si        = ElementPsp(:Si, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
    atoms     = fill(Si, length(positions))

    model  = model_DFT(lattice, atoms, positions;
                       functionals = LDA(), temperature = 0.01)
    basis  = PlaneWaveBasis(model; Ecut = 15, kgrid = kgrid)
    scfres = self_consistent_field(basis; tol = 1e-8, callback = (_) -> nothing)

    kpath = irrfbz_path(model)
    bands = compute_bands(scfres, kpath; kline_density = 20u"bohr")

    return extract_k_and_eigenvalues(bands; keep_bands)
end

# ---------------------------------------------------------------------------
# Apply a random stretch + rattle perturbation to a crystal structure.
#
# The perturbation is:
#   1. rattle!  — random atomic displacements (relative to original lattice)
#   2. stretch  — uniform anisotropic scaling of the cell and all positions
#
# All position arithmetic is done in Angstrom to avoid unit confusion.
# ---------------------------------------------------------------------------
function perturb(sys; ϵ = 0.1)
    sys_pert = deepcopy(sys)

    # Step 1: rattle atoms in place (relative to original cell)
    rattle!(sys_pert, ϵ)

    # Step 2: build a random anisotropic stretch matrix
    stretch = diagm(1 .+ ϵ * (2rand(3) .- 1))

    # Step 3: stretch cell vectors (strip units, stretch, restore units)
    new_vectors = ntuple(3) do i
        v = ustrip.(u"Å", sys_pert.cell.cell_vectors[i])
        (stretch * v) * u"Å"
    end

    # Step 4: stretch atomic positions consistently
    for i in eachindex(sys_pert.particles)
        p   = sys_pert.particles[i]
        pos = ustrip.(u"Å", p.position)
        sys_pert.particles[i] = _set_position(p, (stretch * pos) * u"Å")
    end

    return AtomsBase.FlexibleSystem(
        sys_pert.particles;
        cell_vectors = new_vectors,
        periodicity  = sys_pert.cell.periodicity,
        data         = sys_pert.data,
    )
end

# ---------------------------------------------------------------------------
# Build (bs, λ_ref_list, G_list) for a list of structures — spherical basis.
#
# Each structure gets its own BasisSet (positions differ after perturbation).
# The BasisSet of structure 1 (equilibrium) is returned for model construction,
# since all structures share the same element / basis-set type and thus the
# same nbas and shell layout.
# ---------------------------------------------------------------------------
function build_ref_and_graph_lists_sph(syslist, rcut, kgrid;
                                        keep_bands = 4,
                                        basis_name = "def2-SVP")
    λ_ref_list = Vector{Any}(undef, length(syslist))
    G_list     = Vector{Any}(undef, length(syslist))
    bs_ref     = nothing   # will be set from the first (equilibrium) structure

    for i in eachindex(syslist)
        sys   = syslist[i]
        nlist = PairList(sys, rcut)
        X_ang = [ustrip.(u"Å", r) for r in position(sys, :)]

        bs_i = BasisSet(basis_name,
                        [Molecules.Atom(14, 0.0, X_ang[j]) for j in eachindex(X_ang)];
                        spherical = true, lib = :acsint)

        # keep the equilibrium BasisSet for model construction
        if i == 1
            bs_ref = bs_i
        end

        println("==== [$i] generate_data ===="); flush(stdout)
        klist, λ_ref = generate_data(sys, kgrid; keep_bands = keep_bands)

        println("==== [$i] build ETGraph ===="); flush(stdout)
        G = RealMLTB.nlist2graph(bs_i, nlist, sys, klist)

        λ_ref_list[i] = λ_ref
        G_list[i]     = G
        println("==== [$i] done ===="); flush(stdout)
    end

    return bs_ref, λ_ref_list, G_list
end

# ---------------------------------------------------------------------------
# Build (bs, λ_ref_list, G_list) for a list of structures — Cartesian basis.
# ---------------------------------------------------------------------------
function build_ref_and_graph_lists_cart(syslist, rcut, kgrid;
                                         keep_bands = 4,
                                         basis_name = "def2-SVP")
    λ_ref_list = Vector{Any}(undef, length(syslist))
    G_list     = Vector{Any}(undef, length(syslist))
    bs_ref     = nothing

    for i in eachindex(syslist)
        sys   = syslist[i]
        nlist = PairList(sys, rcut)
        X_ang = [ustrip.(u"Å", r) for r in position(sys, :)]

        bs_i = BasisSet(basis_name,
                        [Molecules.Atom(14, 0.0, X_ang[j]) for j in eachindex(X_ang)];
                        spherical = false, lib = :acsint)

        if i == 1
            bs_ref = bs_i
        end

        println("==== [$i] generate_data ===="); flush(stdout)
        klist, λ_ref = generate_data(sys, kgrid; keep_bands = keep_bands)

        println("==== [$i] build ETGraph ===="); flush(stdout)
        G = RealMLTB.nlist2graph(bs_i, nlist, sys, klist)

        λ_ref_list[i] = λ_ref
        G_list[i]     = G
        println("==== [$i] done ===="); flush(stdout)
    end

    return bs_ref, λ_ref_list, G_list
end