function nlist2graph_cut(bs_name, nlist, sys, klist; 
                         ρcuts=1:0.2:nlist.cutoff, 
                         cutoff=1e-6)
    # for each distinct chemical species in the system,
    # generate a GTO basis centered at 0 
    # the coefficient matrix and cartisan S matrix , 
    # used to transform the Cartesian GTO into spherical GTO, 
    # the onsite T, S, V matrices
    bs_list = Dict{ChemicalSpecies,BasisSet}()
    cart2sph_list = Dict{ChemicalSpecies,Matrix{Float64}}()
    Scart_list = Dict{ChemicalSpecies,Matrix{Float64}}()
    T_on = Dict{ChemicalSpecies,Matrix{Float64}}()
    S_on = Dict{ChemicalSpecies,Matrix{Float64}}()
    V_on = Dict{ChemicalSpecies,Array{Float64,3}}()
    𝐫0 = SA[0.0, 0.0, 0.0]
    for p in sys.particles
        el = species(p)
        if !haskey(bs_list, el)
            Z = Int64(AtomsBase.atomic_number(el))
            mass = ustrip(AtomsBase.mass(el))
            at = [Molecules.Atom(Z, mass, 𝐫0)]
            bs = BasisSet(bs_name, at; spherical=false, lib=:acsint)
            C = RealMLTB.cart2sph_coeff(bs)
            S_cart = overlap(bs)
        
            T_sph = one2c_spherical_AO_integrals(kinetic(bs), C, C, S_cart, S_cart)
            S_sph = one2c_spherical_AO_integrals(S_cart, C, C, S_cart, S_cart)
            V_sph = v3c_spherical_AO_integrals(EPI_1e3c(bs), C, C, C, S_cart, S_cart, S_cart)

            bs_list[el] = bs
            cart2sph_list[el] = C
            Scart_list[el] = S_cart
            T_on[el] = T_sph
            S_on[el] = S_sph
            V_on[el] = V_sph
        end
    end

    X = position(sys, :)
    ii = copy(nlist.i)
    jj = copy(nlist.j)
    first = copy(nlist.first)
    num_atoms = length(first) - 1

    npairs = length(ii)
    Rvec_list = [NeighbourLists._getR(nlist, e) for e in 1:npairs]
    dist_list = norm.(Rvec_list)
    z0_list = [NeighbourLists.AtomsBase.species(sys, i) for i in ii]
    z1_list = [NeighbourLists.AtomsBase.species(sys, j) for j in jj]

    Rcell_list = Vector{SVector{3,Float64}}(undef, npairs)
    for e in 1:npairs
        R_vec = Rvec_list[e]
        R_cell = (R_vec - UnitfulAtomic.ustrip.(X[jj[e]] - X[ii[e]])) .* RealMLTB.ang2bohr
        Rcell_list[e] = SVector{3,Float64}(R_cell)
    end

    println("Computing $npairs pair integrals ...")
    flush(stdout)

    function edge_2c_data(fun2c::Function)
        V2list = Vector{Matrix{Float64}}(undef, npairs)
        Threads.@threads for e in 1:npairs
            Rvec = Rvec_list[e]
            bs1 = bs_list[z0_list[e]]
            bs2 = shift_bs(bs_list[z1_list[e]], Rvec)
            C1 = cart2sph_list[z0_list[e]]
            C2 = cart2sph_list[z1_list[e]]
            S1 = Scart_list[z0_list[e]]
            S2 = Scart_list[z1_list[e]]

            V2_cart = fun2c(bs1, bs2)
            V2_sph = one2c_spherical_AO_integrals(V2_cart, C1, C2, S1, S2)

            V2list[e] = V2_sph
        end
        errs = [cal_err(V2list, dist_list, ρcut) for ρcut in ρcuts]
        idx = findfirst(x -> x < cutoff, errs)
        actual_rcut = isnothing(idx) ? ρcuts[end] : ρcuts[idx]
        fcut(x) = fun_cut(x; rcut=actual_rcut, hw=0.2)

        edge_data = [
            (
                integral_idx=(ii[e], jj[e]),
                Rvec=Rvec_list[e], 
                z=(z0_list[e], z1_list[e]),
                R_cell=Rcell_list[e],
                w=fcut(dist_list[e]),
                V2=V2list[e]
            )
            for e in 1:npairs if fcut(dist_list[e]) > 1e-14
        ]

        return edge_data
    end
    edge_T_data = edge_2c_data(kinetic)
    edge_S_data = edge_2c_data(overlap)

    function edge_3c_data()
        max_cutoff = maximum(ρcuts)
        atom_species = species.(sys.particles)

        tasks = Tuple{Int,Int,Int}[]
        for i in 1:num_atoms
            start_idx = first[i]
            end_idx = first[i+1] - 1
            for idx_j in start_idx:end_idx
                Rij = Rvec_list[idx_j]
                for idx_k in start_idx:end_idx
                    Rik = Rvec_list[idx_k]
					if cal_dist_3c(Rij, Rik) <= max_cutoff
                        push!(tasks, (i, idx_j, idx_k))
                    end
                end
            end
        end

        ntasks = length(tasks)
        println("Total 3-center tasks after distance pre-cut: $ntasks")

        nthreads = Threads.nthreads()
        v_buf = [Array{Float64,3}[] for _ in 1:nthreads]
        z_buf = [Tuple{ChemicalSpecies,ChemicalSpecies,ChemicalSpecies}[] for _ in 1:nthreads]
        idx_buf = [Tuple{Int,Int,Int}[] for _ in 1:nthreads]
        rvec_buf = [Tuple{SVector,SVector}[] for _ in 1:nthreads]
        rcell_idx_buf = [Int[] for _ in 1:nthreads]

		Threads.@threads for t in 1:ntasks
        	tid = Threads.threadid()
        	i, idx_j, idx_k = tasks[t]

            eli, elj, elk = atom_species[i], z1_list[idx_j], z1_list[idx_k]
            Rij, Rik = Rvec_list[idx_j], Rvec_list[idx_k]

            bsi = bs_list[eli]
            bsj = shift_bs(bs_list[elj], Rij)
            bsk = shift_bs(bs_list[elk], Rik)

            V_cart = EPI_1e3c(bsi, bsj, bsk)
            V_sph = v3c_spherical_AO_integrals(V_cart,
                cart2sph_list[eli], cart2sph_list[elj], cart2sph_list[elk],
                Scart_list[eli], Scart_list[elj], Scart_list[elk])

            push!(v_buf[tid], V_sph)
            push!(z_buf[tid], (eli, elj, elk))
            push!(idx_buf[tid], (i, jj[idx_j], jj[idx_k]))
            push!(rvec_buf[tid], (Rij, Rik))
            push!(rcell_idx_buf[tid], idx_k)
        end

        Vlist = vcat(v_buf...)
        zlist = vcat(z_buf...)
        triplet_idx = vcat(idx_buf...)
        Rvec_pairs = vcat(rvec_buf...)
        Rcell_idx = vcat(rcell_idx_buf...)

        dist_3c_list = cal_dist_3c(Rvec_pairs)
        errs = [cal_err(Vlist, dist_3c_list, ρcut) for ρcut in ρcuts]
        idx = findfirst(x -> x < cutoff, errs)
        actual_rcut = isnothing(idx) ? ρcuts[end] : ρcuts[idx]
        fcut(x) = fun_cut(x; rcut=actual_rcut, hw=0.2)

        ntriplets = length(Rvec_pairs)
        edge_data = [
            (
                integral_idx=triplet_idx[e],
                Rpair=Rvec_pairs[e],
                z=zlist[e],
                R_cell=Rcell_list[Rcell_idx[e]],
                w=fcut(dist_3c_list[e]),
                V3=Vlist[e]
            )
            for e in 1:ntriplets if fcut(dist_3c_list[e]) > 1e-14
        ]
        return edge_data
    end
    edge_V_data = edge_3c_data()
    
    klist_phys = physical_k.(Ref(sys), klist)

    G = EquivariantTensors.ETGraph(
        ii, jj;
        edge_data=(
			T_off = edge_T_data,
        	S_off = edge_S_data,
        	V_off = edge_V_data,
		),
        graph_data=(
            klist=klist_phys,
            T_on=T_on,
            V_on=V_on,
            S_on=S_on,
        ),
    )

    @assert G.first == first
    return G
end

function one2c_spherical_AO_integrals(V2_cart, C1, C2, S11_cart, S22_cart)
    V2_sph_raw = C1' * V2_cart * C2

    d1 = get_d_factor(C1, S11_cart)
    d2 = get_d_factor(C2, S22_cart)

    V2_sph = V2_sph_raw .* (d1 * d2')

    return V2_sph
end

function v3c_spherical_AO_integrals(V_cart, C1, C2, C3, S11_cart, S22_cart, S33_cart)
    d1 = get_d_factor(C1, S11_cart)
    d2 = get_d_factor(C2, S22_cart)
    d3 = get_d_factor(C3, S33_cart)

    n1_s = size(C1, 2)
    n2_s = size(C2, 2)
    n3_s = size(C3, 2)

    V_tmp1 = C1' * reshape(V_cart, size(C1, 1), :)
    V_p1 = permutedims(reshape(V_tmp1, n1_s, size(C2, 1), size(C3, 1)), (2, 3, 1))

    V_tmp2 = C2' * reshape(V_p1, size(C2, 1), :)
    V_p2 = permutedims(reshape(V_tmp2, n2_s, size(C3, 1), n1_s), (2, 3, 1))

    V_tmp3 = C3' * reshape(V_p2, size(C3, 1), :)
    V_sph_raw = permutedims(reshape(V_tmp3, n3_s, n1_s, n2_s), (2, 3, 1))

    V_sph = V_sph_raw .* reshape(d1, :, 1, 1) .* reshape(d2, 1, :, 1) .* reshape(d3, 1, 1, :)

    return V_sph
end

function get_d_factor(C, S_cart)
    d = 1.0 ./ sqrt.(diag(C' * S_cart * C))
    replace!(d, Inf => 0.0, NaN => 0.0)
    return d
end

function cal_err(V_list, dist_list, ρcut)
    err = 0.0
    for (e, d) in enumerate(dist_list)
        if d >= ρcut
            tmp = maximum(abs, V_list[e])
            if tmp ≥ err
                err = tmp
            end
        end
    end

    return err
end

function cal_dist_3c(Rij, Rik)
	d_ij = norm(Rij)
    d_ik = norm(Rik)
    d_jk = norm(Rik .- Rij)

    r = sqrt(d_ij^2 + d_ik^2 + d_jk^2)
end

function cal_dist_3c(Rvec_pairs)
    ntriplets = length(Rvec_pairs)
    triplet_distances = zeros(ntriplets)

    for e in 1:ntriplets
        Rij, Rik = Rvec_pairs[e]
        triplet_distances[e] = cal_dist_3c(Rij, Rik)
    end

    return triplet_distances
end

function fun_cut(r; rcut=6, hw=0.2)
    r1 = rcut - hw
    r2 = rcut + hw
    f(x) = 2x^3 - 3x^2 + 1

    if r < r1
        val = 1.0
    elseif r < r2
        rs = (r - r1) / (r2 - r1)
        val = f(rs)
    else
        val = 0.0
    end

    return val
end

function bs_sph_to_cart(bs_in)
    bs =
        if (bs_in.basis[1] isa SphericalShell) || (bs_in.lib != :acsint)
            BasisSet(bs_in.name, bs_in.atoms; spherical=false, lib=:acsint)
        else
            bs_in
        end

    return bs
end

function shift_bs(B, r)
    new_ats = [Molecules.Atom(at.Z, at.mass, r + at.xyz) for at in B.atoms]
    isspherical = B.basis[1] isa SphericalShell
    Bshift = BasisSet(B.name, new_ats;
        spherical=isspherical, lib=:acsint)

    return Bshift
end
