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

# ---------------------------------------------------------------------------
# Project the Hamiltonian from a completed (DFTK) SCF result onto atomic orbitals.
# ---------------------------------------------------------------------------
function project_pw_to_ao_hamiltonian(scfres, bs_ao::BasisSet)
    scfres = DFTK.unfold_bz(scfres)
    bs_pw = scfres.basis
    nk = length(bs_pw.kpoints)
    projs = build_pw_to_ao_projectors(bs_pw, bs_ao)

    H_AO = sum(1:nk) do ik
        bs_pw.kweights[ik] * projs[ik]' * (scfres.ham.blocks[ik] * projs[ik])
    end

    return H_AO
end

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
                                G_plus_ks::AbstractVector{<:AbstractVector{Vec3{T}}}) where {T}
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
function generate_data_graphene(sys, kgrid; Ecut=15)
    lattice = RealMLTB.get_lattice(sys)
    positions = RealMLTB.fractional_coordinates(sys)
    C = ElementPsp(:C, PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth"))
    atoms = fill(C, length(positions))

    model = model_DFT(lattice, atoms, positions;
                      functionals=PBE(), temperature=1e-3)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; tol=1e-8, callback=(_) -> nothing)

    basis_new = DFTK.unfold_bz(basis)
    k_list = [SVector{3,Float64}(kp.coordinate) for kp in basis_new.kpoints]

    return k_list, scfres
end

function build_graphene()
    L = 20  # height of the simulation box
    temperature = 1e-3

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

    model = model_DFT(lattice, atoms, positions; functionals=PBE(), temperature)
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
function build_ref_and_graph_lists_sph_graphene(syslist, rcut, kgrid;
                                                basis_name = "def2-SVP")
    H_ref_list = Vector{Any}(undef, length(syslist))
    G_list     = Vector{Any}(undef, length(syslist))
    bs_ref     = nothing   # will be set from the first (equilibrium) structure

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
        klist, scfres = generate_data_graphene(sys, kgrid)
        H_ref = project_pw_to_ao_hamiltonian(scfres, bs_i)

        println("==== [$i] build ETGraph ===="); flush(stdout)
        G = RealMLTB.nlist2graph(bs_i, nlist, sys, klist)

        H_ref_list[i] = H_ref
        G_list[i]     = G
        println("==== [$i] done ===="); flush(stdout)
    end

    return bs_ref, H_ref_list, G_list
end
