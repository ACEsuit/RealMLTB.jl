using UnitfulAtomic, Unitful
using DFTK
using StaticArrays
using AtomsBase
using PseudoPotentialData
using GaussianBasis
import GaussianBasis: Acsint
using Unitful: Å
using AtomsBuilder: rattle!
using LinearAlgebra
using RealMLTB

# ---------------------------------------------------------------------------
# Build the matrices of projectors onto the spherical atomic orbitals.
#
#      projector[ik][iG, iϕ] = 1/√Ω FT{ ϕₙₗₘ(. - Rᵢ) }(k+G)
#
# where Ω is the unit cell volume, ϕₙₗₘ(. - Rᵢ) is the spherical atomic orbital 
# (n, l, m) centered at Rᵢ
# ---------------------------------------------------------------------------
function build_pw_to_ao_projectors(bs_pw::PlaneWaveBasis{T}, bs_ao::BasisSet) where {T}
    G_plus_k_all = [Gplusk_vectors(bs_pw, bs_pw.kpoints[ik])
                    for ik = 1:length(bs_pw.kpoints)]
    G_plus_k_all_cart = [map(DFTK.recip_vector_red_to_cart(bs_pw.model), gpk) 
                         for gpk in G_plus_k_all]
    nk = length(G_plus_k_all)
    
    projectors = [Matrix{Complex{T}}(undef, length(G_plus_k), 0) for G_plus_k in G_plus_k_all]
    for B in bs_ao.basis
        form_factors = build_sph_form_factors(B, G_plus_k_all_cart)
        pos = B.atom.xyz .* Acsint.ang2bohr
        for ik in 1:nk
            structure_factor = exp.(-im.*dot.(G_plus_k_all_cart[ik],[pos]))
            projectors[ik] = hcat(projectors[ik],                                  
                                  form_factors[ik] .* structure_factor 
                                  ./ sqrt(bs_pw.model.unit_cell_volume))
        end
    end

    return projectors
end

# ---------------------------------------------------------------------------
# Build Fourier transform factors of a spherical basis function centered at 0.
# ---------------------------------------------------------------------------
function build_sph_form_factors(B::SphericalShell, 
                                G_plus_ks::AbstractVector)
    fun(q) = eval_sph_fun_fourier(B, q)
    DFTK.build_form_factors(fun, B.l, G_plus_ks)
end

function eval_sph_fun_fourier(B::SphericalShell, q::T) where {T<:Real}
    f(r) = sum(zip(B.coef, B.exp)) do (ca, a)
        ca*exp(-a*r^2)
    end

    amin = minimum(B.exp)
    R_cut = 3maximum(B.coef) / amin
    h = 0.01 / amin
    rs = 0:h:R_cut
    r2_sphf = rs .^ (2+B.l) .* f.(rs)
    
    DFTK.hankel(rs, r2_sphf, B.l, q)
end

# ---------------------------------------------------------------------------
# Run DFT on sys and return k-points and scfres
# ---------------------------------------------------------------------------
function generate_data_graphene(sys, kgrid, filter::FermiFilter, 
                                bs_ao::BasisSet; Ecut=15, 
                                nbands_plus=10, temperature=1e-3, 
                                kline_density=15u"bohr",
                                cone_scale=3)
    lattice = RealMLTB.get_lattice(sys)
    positions = RealMLTB.fractional_coordinates(sys)
    C = ElementPsp(:C, PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth"))
    atoms = fill(C, length(positions))

    model = model_DFT(lattice, atoms, positions;
                      functionals=PBE(), temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; tol=1e-8, callback=(_) -> nothing)

    εF = scfres.εF
    filter_new = FermiFilter(εF, filter.σ, filter.width, inv(temperature))
    if minimum(x->x[end], scfres.eigenvalues) >= εF + filter.width
        nbands_plus = 0
    end
    nbands = AdaptiveBands(basis.model).n_bands_compute + nbands_plus

    λs, ψs = extract_eigen(scfres.basis, nbands; ρ=scfres.ρ)

    k_list = [SVector{3,Float64}(kp.coordinate) for kp in basis.kpoints]
    proj = build_pw_to_ao_projectors(basis, bs_ao)

    # Extract information near the Dirac (K) point
    kpath = irrfbz_path(model; dim=2, space_group_number=13)
    kinter = DFTK.Brillouin.interpolate(kpath, density=austrip(kline_density))
    kcoords = [SVector{3,Float64}(x...,0.0) for x in kinter.kpaths[1]]
    K = SVector{3,Float64}(kpath.points[:K]..., 0.0)
    M = SVector{3,Float64}(kpath.points[:M]..., 0.0)
    dist = norm(basis.model.recip_lattice * (K-M)) / cone_scale
    Kind = findall(x -> norm(basis.model.recip_lattice * (K-x)) < dist, kcoords)
    bs_K = PlaneWaveBasis(basis, ExplicitKpoints(vcat(kcoords[Kind],[-K])))
    λs_K, ψs_K = extract_eigen(bs_K, nbands; ρ=scfres.ρ)

    append!(k_list, kcoords[Kind], [-K])
    append!(λs, λs_K)
    append!(ψs, ψs_K)    
 
    return k_list, basis.kweights, proj, filter_new, λs, ψs
end

function extract_eigen(basis, nbands::Int; kws...)
    ham = Hamiltonian(basis; kws...)
    eigres = diagonalize_all_kblocks(lobpcg_hyper, ham, nbands; ψguess=nothing)
    λs = [λ[1:nbands] for λ in eigres.λ]
    ψs = [X[:,1:nbands] for X in eigres.X]

    return λs, ψs
end

function build_graphene()
    L = 20  # height of the simulation box

    # Define the geometry and pseudopotential
    a = 4.66  # lattice constant
    a1 = a * [1 / 2, -sqrt(3) / 2, 0]
    a2 = a * [1 / 2, sqrt(3) / 2, 0]
    a3 = L * [0, 0, 1]
    lattice = [a1 a2 a3]
    C1 = [1 / 3, -1 / 3, 0.0]  # in reduced coordinates
    C2 = -C1
    positions = [C1, C2]
    C = ElementPsp(:C, PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth"))
    atoms = [C, C]

    model = model_DFT(lattice, atoms, positions; functionals=PBE())
    sys_bohr = DFTK.periodic_system(model)

    cell_ang = [ustrip.(u"Å", cv) * u"Å" for cv in sys_bohr.cell.cell_vectors]
    atoms_ang = map(enumerate(sys_bohr.particles)) do (i, atom)
        pos_ang = ustrip.(u"Å", atom.position) * u"Å"
        AtomsBase.Atom(species(atom), pos_ang; mass=AtomsBase.mass(atom))
    end
    sys_ang = AtomsBase.FlexibleSystem(
        atoms_ang;
        cell_vectors=cell_ang,
        periodicity=sys_bohr.cell.periodicity,
        data=sys_bohr.data,
    )

    return sys_ang
end

# ---------------------------------------------------------------------------
# Build (bs, H_ref_list, G_list) for a list of structures — spherical basis.
#
# Each structure gets its own BasisSet (positions differ after perturbation).
# The BasisSet of structure 1 (equilibrium) is returned for model construction,
# since all structures share the same element / basis-set type and thus the
# same nbas and shell layout.
# ---------------------------------------------------------------------------
function build_ref_and_graph_lists_sph_graphene(syslist, rcut, kgrid, filter::FermiFilter;
                                                basis_name = "def2-SVP", kws...)
    λ_ref_list  = Vector{Any}(undef, length(syslist))
    ψ_ref_list  = Vector{Any}(undef, length(syslist))
    filter_list = Vector{Any}(undef, length(syslist))
    proj_list   = Vector{Any}(undef, length(syslist))
    kw_list     = Vector{Any}(undef, length(syslist))    
    G_list      = Vector{Any}(undef, length(syslist))
    bs_ref      = nothing   # will be set from the first (equilibrium) structure

    for i in eachindex(syslist)
        sys   = syslist[i]
        nlist = PairList(sys, rcut)
        X_ang = [ustrip.(u"Å", r) for r in position(sys, :)]

        Z = Int64.([AtomsBase.atomic_number(species(p)) for p in sys.particles])
        mass = ustrip.([AtomsBase.mass(species(p)) for p in sys.particles])
        bs_i = BasisSet(basis_name,
                        [Molecules.Atom(Z[j], mass[j], X_ang[j]) 
                                        for j in eachindex(X_ang)];
                        spherical = true, lib = :acsint)

        # keep the equilibrium BasisSet for model construction
        if i == 1
            bs_ref = bs_i
        end

        println("==== [$i] generate_data ===="); flush(stdout)
        klist, kweights, proj, f_i, λs, ψs = generate_data_graphene(sys, kgrid, filter, bs_i; kws...)

        println("==== [$i] build ETGraph ===="); flush(stdout)
        G = RealMLTB.nlist2graph_cut(bs_i, nlist, sys, klist)

        λ_ref_list[i]  = λs
        ψ_ref_list[i]  = ψs
        filter_list[i] = f_i
        proj_list[i]   = proj
        kw_list[i]     = kweights
        G_list[i]      = G
        println("==== [$i] done ===="); flush(stdout)
    end

    return bs_ref, λ_ref_list, ψ_ref_list, filter_list, proj_list, kw_list, G_list
end
