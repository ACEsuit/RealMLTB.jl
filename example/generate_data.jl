using UnitfulAtomic, Unitful
using DFTK
using StaticArrays
using AtomsBase
using PseudoPotentialData

function get_lattice(sys)
    return austrip.(stack(cell_vectors(sys)))
end

function fractional_coordinates(sys)
    lattice = austrip.(stack(cell_vectors(sys)))
    positions = map(sys) do atom
        coordinate = zeros(Float64, 3)
        coordinate[1:3] = lattice[1:3, 1:3] \ Float64.(austrip.(position(atom)))
        SVector{3, Float64}(coordinate)
    end
    return positions
end

function reciprocal_lattice(lattice)
    return 2π * SMatrix{3,3}(inv(lattice)')
end

function physical_k(sys, k_reduced::SVector{3, <:Real})
    lattice = get_lattice(sys)
    G_basis = reciprocal_lattice(lattice)
    return G_basis * k_reduced
end

function extract_k_and_eigenvalues(basis, scfres; keep_bands::Int = 4)
    k_list = [SVector{3,Float64}(kp.coordinate) for kp in basis.kpoints]
    λ_matrix = stack(scfres.eigenvalues)
    λ_targets = [sort(λ_matrix[1:keep_bands, i]) for i in eachindex(k_list)]
    return k_list, λ_targets
end

function generate_data(sys, kgrid; keep_bands::Int = 4)
    dataset = Vector{Tuple{Any, SVector{keep_bands, Float64}}}()

    lattice = get_lattice(sys)
    positions = fractional_coordinates(sys)
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

const ANGSTROM_TO_BOHR = ustrip(Unitful.uconvert(u"bohr", 1u"Å") )

function nlist2graph(bs, nlist, sys, klist)
   X = position(sys, :)
   ii = copy(nlist.i)
   jj = copy(nlist.j)
   first = copy(nlist.first) 
   R_ij = [ NeighbourLists._getR(nlist, n) for n = 1:length(ii) ] 
   S_i = [ NeighbourLists.AtomsBase.species(sys, i) for i in ii ] 
   S_j = [ NeighbourLists.AtomsBase.species(sys, j) for j in jj ]

   Rcell = zeros(SVector{3, Float64}, first[end] - 1)
   Tlist = []
   Vlist = []
   Slist = []
   idx = 0
   for n = 1:length(sys)
      neigs, Rs = NeighbourLists.neigs(nlist, n)
      for m = 1:length(neigs)
         idx += 1
         Rcell[idx] = (Rs[m] - ustrip.(X[neigs[m]] - X[n])) .* ANGSTROM_TO_BOHR
         T_pair, V_pair, S_pair = RealMLTB.compute_pair_AO_integrals(bs, Rcell[idx])
         push!(Tlist, T_pair)
         push!(Vlist, V_pair)
         push!(Slist, S_pair)
      end
   end
   X_ij = [ (𝐫 = 𝐫, s0 = si, s1 = sj, rij = rij, t = t, v = v, s = s) for (𝐫, si, sj, rij, t, v, s) in zip(R_ij, S_i, S_j, Rcell, Tlist, Vlist, Slist) ]
   
   klist = physical_k.(Ref(sys), klist)
   G = EquivariantTensors.ETGraph(ii, jj; edge_data = X_ij, node_data = klist)
   @assert G.first == first

   return G 
end 
