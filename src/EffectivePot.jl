using GaussianBasis
import GaussianBasis: num_basis

const ang2bohr = GaussianBasis.Acsint.ang2bohr

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

# Effective potential integral over three centers
function EPI_1e3c(BS::BasisSet)
    out = zeros(BS.nbas, BS.nbas, BS.nbas)
    EPI_1e3c!(out, BS)
end

function EPI_1e3c!(out, BS::BasisSet)
    Nvals = num_basis.(BS.basis)
    Nmax = maximum(Nvals)

    # Offset list for each shell, used to map shell index to AO index
    ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:BS.nshells]

    ijk = find_unique_ijk(BS.nshells)

    allocate(body) = body(zeros(Cdouble, Nmax^3))
    GaussianBasis.workerpool(allocate, ijk; chunksize=10) do (i, j, k), buf
        @inbounds begin
            Ni = Nvals[i]
            Nj = Nvals[j]
            Nk = Nvals[k]
            Nijk = Ni*Nj*Nk
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
                for l1 in am1 : -1 : 0, n1 in 0 : am1 - l1
                    m1 = am1 - l1 - n1
                    index2 = 1
                    for l2 in am2 : -1 : 0, n2 in 0 : am2 - l2
                        m2 = am2 - l2 - n2
                        index3 = 1
                        for l3 in am3 : -1 : 0, n3 in 0 : am3 - l3
                            m3 = am3 - l3 - n3
                            index = index1 + N1*(index2-1) + N1*N2*(index3-1)
                            out[index] += E3[1, 1, l1+1, l2+1, l3+1] * E3[2, 1, m1+1, m2+1, m3+1] * E3[3, 1, n1+1, n2+1, n3+1] * prefac
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

@inline function generate_E3_matrix!(E3, maxam1, maxam2, maxam3,
                           P, Q, A, B, C,
                           a::Float64, b::Float64, c::Float64)
    # Makes the Hermite->Cartesian conversion factors by recursion
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

    @inbounds @. E3[:, 1, 1, 1, 1] = exp(-µ*AB^2-ν*PC^2)
    oo2q = 1 / (2 * q)
    @inbounds for k = 1 : maxam3 + 1, j = 1 : maxam2 + 1, i = 1 : maxam1 + 1
      # handle t = 1 case
      if i > 1
         @views E3[:, 1, i, j, k] .+= QA .* E3[:, 1, i - 1, j, k] .+ E3[:, 2, i - 1, j, k]
      elseif j > 1
         @views E3[:, 1, i, j, k] .+= QB .* E3[:, 1, i, j - 1, k] .+ E3[:, 2, i, j - 1, k]
      elseif k > 1
         @views E3[:, 1, i, j, k] .+= QC .* E3[:, 1, i, j, k - 1] .+ E3[:, 2, i, j, k - 1]
      end
      for t = 2 : i + j + k - 1
         if i > 1
            @. @views E3[:, t, i, j, k] += QA * E3[:, t, i - 1, j, k] + t * E3[:, t + 1, i - 1, j, k] + oo2q * E3[:, t - 1, i - 1, j, k]
         elseif j > 1
            @. @views E3[:, t, i, j, k] += QB * E3[:, t, i, j - 1, k] + t * E3[:, t + 1, i, j - 1, k] + oo2q * E3[:, t - 1, i, j - 1, k]
         elseif k > 1
            @. @views E3[:, t, i, j, k] += QC * E3[:, t, i, j, k - 1] + t * E3[:, t + 1, i, j, k - 1] + oo2q * E3[:, t - 1, i, j, k - 1]
         end
      end
   end
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
        ex2 = :(
            for $ii = $start:N
                $ex2
            end
        )
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

using Molecules

function compute_pair_AO_integrals(bs, R_bohr)
    molR_list = [bs.atoms...]
    nbas = bs.nbas
    for atoms in bs.atoms
        atoms_R = Molecules.Atom(atoms.Z, atoms.mass, atoms.xyz + R_bohr ./ ang2bohr)
        push!(molR_list, atoms_R)
    end

    bs_pair = BasisSet(bs.name, molR_list; spherical=false, lib=:acsint)
    S_pair = overlap(bs_pair)[1:nbas, nbas+1:2*nbas]
    T_pair = kinetic(bs_pair)[1:nbas, nbas+1:2*nbas]

    out = zeros(bs_pair.nbas, bs_pair.nbas, bs_pair.nbas)
    EPI_1e3c!(out, bs_pair)
    V_pair = out[1:nbas, 1:nbas, nbas+1:2*nbas]
    return T_pair, V_pair, S_pair
end

function compute_AO_integrals(bs)
    S_pair = overlap(bs)
    T_pair = kinetic(bs)

    V_pair = zeros(bs.nbas, bs.nbas, bs.nbas)
    EPI_1e3c!(V_pair, bs)
    return T_pair, V_pair, S_pair
end

