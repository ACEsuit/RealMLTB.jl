import EquivariantTensors: PState
using EquivariantTensors

function nlist2graph_cut(bs::BasisSet, nlist, sys, klist; kwargs...)
    return nlist2graph_cut(bs.name, nlist, sys, klist; kwargs...)
end

function nlist2graph_cut(bs_name, nlist, sys, klist; 
                         ρcuts=1:0.2:nlist.cutoff, 
                         cutoff=1e-6,
                         rcut_3c_scale=sqrt(3),
                         cs_screen=true,
                         two_pass=true,
                         subsample=400)
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

    # Cauchy-Schwarz scale per pair-list edge, for screening triplets BEFORE
    # their three-centre integral is computed.
    #
    #   |∫ φ_a φ_b φ_c| <= ||φ_a φ_b||_2 ||φ_c||_2 = sqrt(∫ φ_a² φ_b²)
    #
    # since the spherical AOs are normalised.  generate_S2_pair! is exactly
    # ∫ φ_a² φ_b², and sparseEPI_1e3c already screens with its square root, so
    # this is the same bound.  The distance screen alone has to be loose, as
    # actual_rcut is only known from cal_err after the integrals exist: at
    # rcut = 6 Å on 2-atom Si it pushed 4260 triplet tasks through EPI_1e3c and
    # then multiplied 1080 of them by w = 0.  A rigorous bound drops those
    # before the integral rather than after.
    s2_edge = let
        bsmax = maximum(maximum(num_basis(sh) for sh in bs_list[el].basis)
                        for el in keys(bs_list))
        out = zeros(npairs)
        tmp = zeros(Cdouble, bsmax * bsmax)
        for e in 1:npairs
            bs1 = bs_list[z0_list[e]]
            bs2 = shift_bs(bs_list[z1_list[e]], Rvec_list[e])
            m = 0.0
            for sh1 in bs1.basis, sh2 in bs2.basis
                generate_S2_pair!(tmp, sh1, sh2)
                m = max(m, maximum(tmp))
            end
            out[e] = sqrt(max(m, 0.0))
        end
        out
    end
    # the p = i or ket = i home-cell case: both functions on the same atom
    s2_onsite = Dict{ChemicalSpecies,Float64}()
    for (el, bsel) in bs_list
        tmp = zeros(Cdouble, maximum(num_basis(sh) for sh in bsel.basis)^2)
        m = 0.0
        for sh1 in bsel.basis, sh2 in bsel.basis
            generate_S2_pair!(tmp, sh1, sh2)
            m = max(m, maximum(tmp))
        end
        s2_onsite[el] = sqrt(max(m, 0.0))
    end

    function edge_3c_data()
        # The screen coordinate is cal_dist_3c = sqrt(d_ij^2+d_ik^2+d_jk^2),
        # which is sqrt(2)..sqrt(3) times a single pairwise distance for the
        # configurations that matter.  Comparing it against the PAIR cutoff
        # (maximum(ρcuts)) therefore truncated the potential sum far inside
        # the range used for T and S: with rcut = 6 Å the potential reached
        # only 3.84 Å between bra and ket while T and S reached 5.92 Å, and
        # even at nearest-neighbour separation ~42% of the three-centre
        # integral mass was dropped.  Scale the grid the screen and the
        # adaptive cutoff search live on.
        ρcuts_3c = range(Base.first(ρcuts), rcut_3c_scale * maximum(ρcuts);
                         step=step(ρcuts))
        max_cutoff = maximum(ρcuts_3c)
        atom_species = species.(sys.particles)

        # A matrix element of the periodic potential
        #   V(r) = Σ_{p,R'} Σ_b Wv[p,b] φ_{p,b}(r - R_p - R')
        # needs, for every block (i, j, R),
        #   Σ_{p,R'}  ∫ φ_i(r) φ_{p,b}(r - R_p - R') φ_j(r - R_j - R)
        # i.e. BOTH the potential centre and the ket run over atoms AND their
        # images.  The pair list supplies images of OTHER atoms but has no
        # (i, i, R = 0) edge, so a loop that takes both the potential centre
        # and the ket from the pair list drops every term in which one of
        # them is atom i in the home cell.  Two sentinels restore those:
        #
        #   idx_k == 0  ->  ket = atom i, home cell.  Without it the R = 0
        #                   on-site block saw only the potential of atom i
        #                   itself (the whole Σ_{p≠i} crystal field was gone).
        #   idx_j == 0  ->  potential centre = atom i, home cell.  Without it
        #                   EVERY off-diagonal block was missing the largest
        #                   three-centre term there is, the one whose
        #                   potential sits on the bra atom.
        #
        #   (idx_j, idx_k) == (0, 0) is the on-site V_on path and is already
        #   handled there, so it is not emitted here.
        #
        # A fourth family is needed because the potential centre must not be
        # tied to the BRA's neighbourhood at all.  A centre within rcut of the
        # ket but further than rcut from the bra is invisible to nbrs(i); it is
        # generated only by the ket-anchored triplet, lands in the transposed
        # block, and assembly.jl's  H .= (H+H')/2  then keeps HALF of it.  So
        # the potential centre also runs over the KET's neighbours, reached as
        #       R_ip = R_ik + e,      e in nbrs(ket atom),
        # and deduplicated against the bra side by  |R_ip| > pair cutoff
        # (the pair list is complete below that, so anything closer is already
        # emitted as a bra-side centre, and |R_ip| = 0 is the idx_j == 0 case).
        #
        #   idx_e != 0  ->  potential centre taken from the ket's neighbours.
        #
        # With this family in place the p-set of block (i,j,R) equals that of
        # block (j,i,-R), so the raw accumulation is Hermitian and (H+H')/2 in
        # assembly.jl is only a rounding safeguard.
        zeroR = zero(Rvec_list[1])
        rcut_pair = nlist.cutoff
        tasks = NTuple{4,Int}[]
        for i in 1:num_atoms
            start_idx = first[i]
            end_idx = first[i+1] - 1
            son = s2_onsite[atom_species[i]]
            for idx_j in start_idx:end_idx
                Rij = Rvec_list[idx_j]
                if cal_dist_3c(Rij, zeroR) <= max_cutoff &&
                   (!cs_screen || min(s2_edge[idx_j], son) > cutoff)
                    push!(tasks, (i, idx_j, 0, 0))
                end
                for idx_k in start_idx:end_idx
                    Rik = Rvec_list[idx_k]
                    if cal_dist_3c(Rij, Rik) <= max_cutoff &&
                       (!cs_screen || min(s2_edge[idx_j], s2_edge[idx_k]) > cutoff)
                        push!(tasks, (i, idx_j, idx_k, 0))
                    end
                end
            end
            for idx_k in start_idx:end_idx
                Rik = Rvec_list[idx_k]
                if cal_dist_3c(zeroR, Rik) <= max_cutoff &&
                   (!cs_screen || min(son, s2_edge[idx_k]) > cutoff)
                    push!(tasks, (i, 0, idx_k, 0))
                end
                m = jj[idx_k]
                for idx_e in first[m]:first[m+1]-1
                    Rip = Rik + Rvec_list[idx_e]
                    norm(Rip) <= rcut_pair + 1e-10 && continue   # bra side already has it
                    # the bra-to-p pair is not a pair-list edge here, so use the
                    # two bounds that are available; min of a subset is still a
                    # valid upper bound, just weaker.
                    if cal_dist_3c(Rip, Rik) <= max_cutoff &&
                       (!cs_screen || min(s2_edge[idx_k], s2_edge[idx_e]) > cutoff)
                        push!(tasks, (i, 0, idx_k, idx_e))
                    end
                end
            end
        end

        println("3c candidates after distance + Cauchy-Schwarz pre-cut: $(length(tasks))")
        flush(stdout)

        RType = typeof(Rvec_list[1])

        # One place where a triplet's geometry is resolved and its integral is
        # computed, used by BOTH passes below so they cannot drift apart.
        function _one_triplet(tsk)
            i, idx_j, idx_k, idx_e = tsk
            eli = atom_species[i]

            if idx_k == 0
                elk, Rik, ket, R_cell = atom_species[i], zeroR, i,
                                        zero(SVector{3,Float64})
            else
                elk, Rik, ket, R_cell = z1_list[idx_k], Rvec_list[idx_k],
                                        jj[idx_k], Rcell_list[idx_k]
            end

            # potential centre: ket's neighbour, atom i itself, or bra's neighbour
            if idx_e != 0
                elj, Rij, pot = z1_list[idx_e], Rik + Rvec_list[idx_e], jj[idx_e]
            elseif idx_j == 0
                elj, Rij, pot = atom_species[i], zeroR, i
            else
                elj, Rij, pot = z1_list[idx_j], Rvec_list[idx_j], jj[idx_j]
            end

            bsi = bs_list[eli]
            bsj = shift_bs(bs_list[elj], Rij)
            bsk = shift_bs(bs_list[elk], Rik)

            V_sph = v3c_spherical_AO_integrals(EPI_1e3c(bsi, bsj, bsk),
                cart2sph_list[eli], cart2sph_list[elj], cart2sph_list[elk],
                Scart_list[eli], Scart_list[elj], Scart_list[elk])
            return V_sph, (eli, elj, elk), (i, pot, ket), (Rij, Rik), R_cell
        end

        _task_dist(tsk) = cal_dist_3c(
            tsk[2] == 0 ? zeroR :
                (tsk[4] != 0 ? Rvec_list[tsk[3]] + Rvec_list[tsk[4]] :
                               Rvec_list[tsk[2]]),
            tsk[3] == 0 ? zeroR : Rvec_list[tsk[3]])

        # ------------------------------------------------------------------
        # Two-pass cutoff estimate.
        #
        # The smooth cutoff radius actual_rcut is chosen by cal_err FROM the
        # computed integrals, so the one-pass code computes every candidate and
        # then multiplies the ones beyond actual_rcut by w = 0.  How much that
        # wastes is basis dependent: on 2-atom Si at rcut = 6 Å, sto-3g
        # computes 4260 and keeps 1308, while def2-SVP keeps all 4260.
        #
        # So estimate actual_rcut from a subsample first and skip what that
        # estimate would zero.  The subsample keeps every candidate in the far
        # tail, since the tail is what decides the cutoff and an estimate that
        # came out too small would skip work that mattered, plus a stride over
        # the rest and two grid steps of margin.  The final actual_rcut is
        # still taken from cal_err on the full kept set, so it can only come
        # out smaller than the estimate, never larger.
        cand_dist = [_task_dist(t) for t in tasks]

        if two_pass && length(tasks) > 4 * subsample
            # The probe set must be chosen from rotation-invariant data only.
            # Selecting by index (sortperm plus a stride) is not: the index
            # order of two graphs differing by a rigid rotation differs when
            # distances tie, which makes rcut_est, the kept set and hence the w
            # weights weakly rotation dependent.  Selecting by distance value
            # gives an invariant multiset, and cal_err sees only distances and
            # |V| values, both invariant.
            dmin, dmax = extrema(cand_dist)
            tail_cut = dmax - 0.15 * (dmax - dmin)
            nwin = max(subsample ÷ 8, 4)
            step_w = (dmax - dmin) / nwin
            halfw = 0.25 * step_w
            centres = [dmin + (i - 0.5) * step_w for i in 1:nwin]
            probe = findall(d -> d >= tail_cut ||
                                 any(abs(d - c) <= halfw for c in centres),
                            cand_dist)
            Vp = Vector{Array{Float64,3}}(undef, length(probe))
            Threads.@threads for q in eachindex(probe)
                Vp[q] = _one_triplet(tasks[probe[q]])[1]
            end
            errs_p = [cal_err(Vp, cand_dist[probe], ρcut) for ρcut in ρcuts_3c]
            ip = findfirst(x -> x < cutoff, errs_p)
            rcut_est = isnothing(ip) ? ρcuts_3c[end] :
                       min(ρcuts_3c[end], ρcuts_3c[ip] + 2 * step(ρcuts_3c))
            keep = findall(d -> d <= rcut_est + 0.2, cand_dist)
            println("  two-pass: probed $(length(probe)), actual_rcut <= " *
                    "$(round(rcut_est, digits=3)), keeping $(length(keep)) of " *
                    "$(length(tasks))")
            tasks = tasks[keep]
        end

        ntasks = length(tasks)
        println("Total 3-center tasks computed: $ntasks")
        flush(stdout)

        Vlist       = Vector{Array{Float64,3}}(undef, ntasks)
        zlist       = Vector{Tuple{ChemicalSpecies,ChemicalSpecies,ChemicalSpecies}}(undef, ntasks)
        triplet_idx = Vector{Tuple{Int,Int,Int}}(undef, ntasks)  # (bra, pot, ket)
        Rvec_pairs  = Vector{Tuple{RType,RType}}(undef, ntasks)
        Rcell_vals  = Vector{SVector{3,Float64}}(undef, ntasks)

        Threads.@threads for t in 1:ntasks
            Vlist[t], zlist[t], triplet_idx[t], Rvec_pairs[t], Rcell_vals[t] =
                _one_triplet(tasks[t])
        end

        dist_3c_list = cal_dist_3c(Rvec_pairs)
        errs = [cal_err(Vlist, dist_3c_list, ρcut) for ρcut in ρcuts_3c]
        idx = findfirst(x -> x < cutoff, errs)
        actual_rcut = isnothing(idx) ? ρcuts_3c[end] : ρcuts_3c[idx]
        println("  3c: ntasks=", ntasks, "  dist_3c in [",
                round(minimum(dist_3c_list), digits=3), ", ",
                round(maximum(dist_3c_list), digits=3), "]  actual_rcut=",
                round(actual_rcut, digits=3), "  grid top=",
                round(ρcuts_3c[end], digits=3),
                "  errs[1]=", round(errs[1], sigdigits=3),
                "  errs[end]=", round(errs[end], sigdigits=3))
        fcut(x) = fun_cut(x; rcut=actual_rcut, hw=0.2)

        ntriplets = length(Rvec_pairs)
        edge_data = [
            (
                integral_idx=triplet_idx[e],
                Rpair=Rvec_pairs[e],
                z=zlist[e],
                R_cell=Rcell_vals[e],
                w=fcut(dist_3c_list[e]),
                V3=Vlist[e]
            )
            for e in 1:ntriplets if fcut(dist_3c_list[e]) > 1e-14
        ]
        return edge_data
    end
    edge_V_data = edge_3c_data()
    
    klist_phys = physical_k.(Ref(sys), klist)

    edge_data = [
        PState(
            R_cell = Rcell_list[e],
            𝐫      = Rvec_list[e],
            z0     = z0_list[e],
            z1     = z1_list[e],
        )
        for e in 1:npairs
    ]

    G = EquivariantTensors.ETGraph(
        ii, jj;
        edge_data = edge_data,
        graph_data = (
            klist = klist_phys,

            T_on = T_on,
            V_on = V_on,
            S_on = S_on,

            atom_species = species.(sys.particles),

            cut_integrals = (
                T_off = edge_T_data,
                S_off = edge_S_data,
                V_off = edge_V_data,
            ),
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

# Largest integral element among the edges that a cutoff at ρcut would DROP,
# used to pick actual_rcut as the first ρcut whose dropped tail is negligible.
#
# Inf, not 0.0, when no sampled edge lies beyond ρcut: the sample is truncated
# by the pair-list radius, which is the very radius this test is meant to
# validate, so "nothing left to look at" is not evidence of convergence.  With
# 0.0 the comparison 0.0 < cutoff always succeeded just past the outermost
# shell the pair list contained, and the smooth cutoff then tapered that
# outermost shell whatever rcut was.  On 2-atom Si / def2-SVP at rcut = 5 Å
# that damped a shell still carrying max|V2| = 0.24 and left the assembled
# overlap indefinite (min eig S(k) = -2.4e-2 at k = (0.137,0.221,0.079)),
# which makes generate_λ_and_vecs throw.  With Inf, findfirst falls through to
# ρcuts[end] and every computed edge keeps full weight.
function cal_err(V_list, dist_list, ρcut)
    found = false
    err = 0.0
    for (e, d) in enumerate(dist_list)
        if d >= ρcut
            found = true
            tmp = maximum(abs, V_list[e])
            if tmp ≥ err
                err = tmp
            end
        end
    end

    return found ? err : Inf
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
