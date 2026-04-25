using GaussianBasis
using Molecules
import GaussianBasis: num_basis
import GaussianBasis: Acsint

const ang2bohr = Acsint.ang2bohr

function EffectivePot(BS::BasisSet, coef::AbstractVector)
    out = zeros(BS.nbas, BS.nbas, BS.nbas)
    EffectivePot!(out, BS, coef)
end

function EffectivePot!(out, BS::BasisSet, coef::AbstractVector)
    N = BS.nbas
    @assert N == length(coef)
    EPI_1e3c!(out, BS)
    reshape(reshape(out, N^2, N) * coef, N, N)
end

function sparseEPI_1e3c(BS1::BasisSet, BS2::BasisSet, cutoff=1e-12;
                        ntasks=Threads.nthreads())
    Nvals1 = num_basis.(BS1.basis)
    Nvals2 = num_basis.(BS2.basis)
    Nmax1 = maximum(Nvals1)
    Nmax2 = maximum(Nvals2)

    ao_offset1 = [sum(Nvals1[1:(i-1)]) for i = 1:BS1.nshells]
    ao_offset2 = [sum(Nvals2[1:(i-1)]) for i = 1:BS2.nshells]

    tmp = zeros(Cdouble, Nmax1 * Nmax2)
    ki_vals = Vector{Int16}[]
    for (s2, bs2) in enumerate(BS2.basis)
        i_vals = Int16[]
        for (s1, bs1) in enumerate(BS1.basis)
            generate_S2_pair!(tmp, bs1, bs2)
            if √maximum(tmp) > cutoff
                push!(i_vals, s1)
            end
        end
        push!(ki_vals, i_vals)
    end

    results_lock = ReentrantLock()
    all_indexes = NTuple{3,Int16}[]
    all_vals    = Cdouble[]

    allocate(body) = body(zeros(Cdouble, Nmax1^2 * Nmax2))
    GaussianBasis.workerpool(allocate, 1:BS2.nshells;
                             chunksize=1, ntasks=ntasks) do k, buf
        local_idx = NTuple{3,Int16}[]
        local_val = Cdouble[]
        @inbounds begin
            Nk = Nvals2[k]
            koff = ao_offset2[k]
            K = (koff+1):(koff+Nk)
            i_vals = ki_vals[k]
            for ip = 1:length(i_vals)
                i = i_vals[ip]
                for jp = ip:length(i_vals)
                    j = i_vals[jp]
                    Ni = Nvals1[i]
                    Nj = Nvals1[j]
                    ioff = ao_offset1[i]
                    joff = ao_offset1[j]
                    I = (ioff+1):(ioff+Ni)
                    J = (joff+1):(joff+Nj)

                    generate_V_triple!(buf, BS1.basis[i], BS1.basis[j], BS2.basis[k])

                    if maximum(abs, buf) > cutoff
                        for (ni, nI) in enumerate(I), (nj, nJ) in enumerate(J)
                            if nI <= nJ
                                for (nk, nK) in enumerate(K)
                                    nijk = Ni * Nj * (nk - 1) + Ni * (nj - 1) + ni
                                    val = buf[nijk]
                                    if abs(val) > cutoff
                                        push!(local_idx, (nI, nJ, nK))
                                        push!(local_val, val)
                                        if nI < nJ
                                            push!(local_idx, (nJ, nI, nK))
                                            push!(local_val, val)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        lock(results_lock) do
            append!(all_indexes, local_idx)
            append!(all_vals, local_val)
        end
    end

    return all_indexes, all_vals
end

function EPI_1e3c(BS1::BasisSet, BS2::BasisSet)
    out = zeros(BS1.nbas, BS1.nbas, BS2.nbas)
    EPI_1e3c!(out, BS1, BS2)
end

function EPI_1e3c!(out, BS1::BasisSet, BS2::BasisSet;
                   ntasks=Threads.nthreads())
    Nvals1 = num_basis.(BS1.basis)
    Nvals2 = num_basis.(BS2.basis)
    Nmax1 = maximum(Nvals1)
    Nmax2 = maximum(Nvals2)

    ao_offset1 = [sum(Nvals1[1:(i-1)]) for i = 1:BS1.nshells]
    ao_offset2 = [sum(Nvals2[1:(i-1)]) for i = 1:BS2.nshells]

    allocate(body) = body(zeros(Cdouble, Nmax1^2 * Nmax2))
    GaussianBasis.workerpool(allocate, 1:BS2.nshells;
                             chunksize=1, ntasks=ntasks) do k, buf
        @inbounds begin
            Nk = Nvals2[k]
            koff = ao_offset2[k]
            K = (koff+1):(koff+Nk)

            for i in 1:BS1.nshells
                for j = i:BS1.nshells
                    Ni = Nvals1[i]
                    Nj = Nvals1[j]
                    Nijk = Ni * Nj * Nk
                    ioff = ao_offset1[i]
                    joff = ao_offset1[j]
                    I = (ioff+1):(ioff+Ni)
                    J = (joff+1):(joff+Nj)

                    generate_V_triple!(buf, BS1.basis[i], BS1.basis[j], BS2.basis[k])

                    vals = reshape(view(buf, 1:Nijk), Ni, Nj, Nk)
                    out[I, J, K] = vals
                    out[J, I, K] = permutedims(vals, [2, 1, 3])
                end
            end
        end
    end

    return out
end

function EPI_1e3c(BS::BasisSet)
    out = zeros(BS.nbas, BS.nbas, BS.nbas)
    EPI_1e3c!(out, BS)
end

function EPI_1e3c!(out, BS::BasisSet)
    Nvals = num_basis.(BS.basis)
    Nmax = maximum(Nvals)
    ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:BS.nshells]
    ijk = find_unique_ijk(BS.nshells)

    allocate(body) = body(zeros(Cdouble, Nmax^3))
    GaussianBasis.workerpool(allocate, ijk; chunksize=10, ntasks=1) do (i, j, k), buf
        @inbounds begin
            Ni = Nvals[i]
            Nj = Nvals[j]
            Nk = Nvals[k]
            Nijk = Ni * Nj * Nk
            ioff = ao_offset[i]
            joff = ao_offset[j]
            koff = ao_offset[k]

            IJK = [(ioff+1):(ioff+Ni),
                   (joff+1):(joff+Nj),
                   (koff+1):(koff+Nk)]

            generate_V_triple!(buf, BS, i, j, k)

            vals = reshape(view(buf, 1:Nijk), Ni, Nj, Nk)
            sym_perms = find_symmetric_perms(i, j, k)
            for sp in sym_perms
                Ip, Jp, Kp = IJK[sp]
                out[Ip, Jp, Kp] = permutedims(vals, sp)
            end
        end
    end
    return out
end

function generate_V_triple!(out, B1::CartesianShell, B2::CartesianShell, B3::CartesianShell)
    out .= 0.0

    am1 = B1.l
    am2 = B2.l
    am3 = B3.l

    A = B1.atom.xyz .* ang2bohr
    B = B2.atom.xyz .* ang2bohr
    C = B3.atom.xyz .* ang2bohr

    N1 = num_basis(B1)
    N2 = num_basis(B2)

    E3 = similar(A, 3, am1 + am2 + am3 + 3, am1 + 1, am2 + 1, am3 + 1)
    E3 .= 0.0

    for (ca, a) in zip(B1.coef, B1.exp)
        for (cb, b) in zip(B2.coef, B2.exp)
            for (cc, c) in zip(B3.coef, B3.exp)
                p = a + b
                P = (a * A + b * B) / p
                q = p + c
                Q = (p * P + c * C) / q
                prefac = ca * cb * cc * (π / q) ^ 1.5

                generate_E3_matrix!(E3, am1, am2, am3, P, Q, A, B, C, a, b, c)

                index1 = 1
                for l1 in am1:-1:0, n1 in 0:am1-l1
                    m1 = am1 - l1 - n1
                    index2 = 1
                    for l2 in am2:-1:0, n2 in 0:am2-l2
                        m2 = am2 - l2 - n2
                        index3 = 1
                        for l3 in am3:-1:0, n3 in 0:am3-l3
                            m3 = am3 - l3 - n3
                            index = index1 + N1*(index2-1) + N1*N2*(index3-1)
                            out[index] += E3[1, 1, l1+1, l2+1, l3+1] *
                                          E3[2, 1, m1+1, m2+1, m3+1] *
                                          E3[3, 1, n1+1, n2+1, n3+1] * prefac
                            index3 += 1
                        end
                        index2 += 1
                    end
                    index1 += 1
                end
            end
        end
    end
    return out
end

function generate_V_triple!(out, BS::BasisSet, s1, s2, s3)
    generate_V_triple!(out, BS.basis[s1], BS.basis[s2], BS.basis[s3])
end

@inline function generate_E3_matrix!(E3, maxam1, maxam2, maxam3, P, Q, A, B, C, a::Float64, b::Float64, c::Float64)
    E3 .= 0
    p = a + b
    q = p + c

    AB = A - B
    PC = P - C
    QA = Q - A
    QB = Q - B
    QC = Q - C

    µ = a * b / p
    ν = p * c / q

    @inbounds @. E3[:, 1, 1, 1, 1] = exp(-µ * AB^2 - ν * PC^2)
    oo2q = 1 / (2 * q)

    @inbounds for k = 1:maxam3+1, j = 1:maxam2+1, i = 1:maxam1+1
        if i > 1
            @views @. E3[:, 1, i, j, k] += QA * E3[:, 1, i-1, j, k] + E3[:, 2, i-1, j, k]
        elseif j > 1
            @views @. E3[:, 1, i, j, k] += QB * E3[:, 1, i, j-1, k] + E3[:, 2, i, j-1, k]
        elseif k > 1
            @views @. E3[:, 1, i, j, k] += QC * E3[:, 1, i, j, k-1] + E3[:, 2, i, j, k-1]
        end
        for t = 2:i+j+k-1
            if i > 1
                @views @. E3[:, t, i, j, k] += QA * E3[:, t, i-1, j, k] + t * E3[:, t+1, i-1, j, k] + oo2q * E3[:, t-1, i-1, j, k]
            elseif j > 1
                @views @. E3[:, t, i, j, k] += QB * E3[:, t, i, j-1, k] + t * E3[:, t+1, i, j-1, k] + oo2q * E3[:, t-1, i, j-1, k]
            elseif k > 1
                @views @. E3[:, t, i, j, k] += QC * E3[:, t, i, j, k-1] + t * E3[:, t+1, i, j, k-1] + oo2q * E3[:, t-1, i, j, k-1]
            end
        end
    end
end

function generate_S2_pair!(out, B1::CartesianShell, B2::CartesianShell)
    out .= 0.0

    am1 = B1.l
    am2 = B2.l

    A = B1.atom.xyz .* ang2bohr
    B = B2.atom.xyz .* ang2bohr

    N1 = num_basis(B1)

    E = similar(A, 3, 2am1 + 2am2 + 2, 2am1 + 1, 2am2 + 1)
    E .= 0.0

    B12 = square(B1)
    B22 = square(B2)

    for (ca, a) in zip(B12.coef, B12.exp)
        for (cb, b) in zip(B22.coef, B22.exp)
            p = a + b
            P = (a * A + b * B) / p
            prefac = ca * cb * (π / p)^1.5

            Acsint.generate_E_matrix!(E, 2am1, 2am2, P, A, B, a, b)

            index1 = 1
            for l1 in am1:-1:0
                for n1 in 0:am1-l1
                    m1 = am1 - l1 - n1
                    index2 = 1
                    for l2 in am2:-1:0
                        for n2 in 0:am2-l2
                            m2 = am2 - l2 - n2
                            out[index1 + N1*(index2-1)] +=
                                E[1, 1, 2l1+1, 2l2+1] *
                                E[2, 1, 2m1+1, 2m2+1] *
                                E[3, 1, 2n1+1, 2n2+1] * prefac
                            index2 += 1
                        end
                    end
                    index1 += 1
                end
            end
        end
    end
    return out
end

function square(B::CartesianShell{A,R}) where {A,R}
    coef = zeros(R, div(length(B.coef) * (length(B.coef)+1), 2))
    exp  = copy(coef)
    k = 1
    for i = 1:length(B.coef)
        for j = i:length(B.coef)
            coef[k] = i == j ? B.coef[i]^2 : 2B.coef[i]*B.coef[j]
            exp[k]  = B.exp[i] + B.exp[j]
            k += 1
        end
    end
    CartesianShell(2B.l, coef, exp, B.atom)
end

@generated function find_unique_ijk(T, N)
    ex = :(unique_ijk = NTuple{3,T}[])
    tupleex = :(i1)
    for i = 2:3
        ii = Symbol("i$i")
        tupleex = :($tupleex..., $ii)
    end

    ex2 = :(push!(unique_ijk, $tupleex))
    for i = 1:3
        ii = Symbol("i$i")
        start = i == 3 ? 1 : Symbol("i$(i+1)")
        ex2 = quote
            for $ii = $start:N
                $ex2
            end
        end
    end
    return :($ex; $ex2; unique_ijk)
end

find_unique_ijk(N) = find_unique_ijk(Int16, N)

function find_symmetric_perms(i, j, k)
    @assert i ≥ j ≥ k

    if i == j == k
        return [[1, 2, 3]]
    elseif i > j > k
        return [[1, 2, 3], [1, 3, 2], [2, 1, 3],
                [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    elseif i == j
        return [[1, 2, 3], [1, 3, 2], [3, 1, 2]]
    elseif j == k
        return [[1, 2, 3], [2, 1, 3], [2, 3, 1]]
    end
end

function compute_pair_AO_integrals(bs, R_bohr; cutoff=1e-12)
    mol_R = similar(bs.atoms)
    for (i, atoms) in enumerate(bs.atoms)
        mol_R[i] = Molecules.Atom(atoms.Z, atoms.mass, atoms.xyz + R_bohr ./ ang2bohr)
    end
    bs_R = BasisSet(bs.name, mol_R; spherical=false, lib=:acsint)

    T_pair = kinetic(bs, bs_R)
    S_pair = overlap(bs, bs_R)
    V_pair = sparseEPI_1e3c(bs, bs_R, cutoff)

    return T_pair, V_pair, S_pair
end

function compute_AO_integrals(bs)
    T = kinetic(bs)
    V = EPI_1e3c(bs)
    S = overlap(bs)
    return T, V, S
end
