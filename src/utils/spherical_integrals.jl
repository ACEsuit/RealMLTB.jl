using LinearAlgebra
using SpecialFunctions
using GaussianBasis
using Molecules
import GaussianBasis: Acsint

const ang2bohr = Acsint.ang2bohr

basis_angular(B::BasisFunction) = B.l

function alt_sign(m::Int)
    isodd(m) ? -1 : 1
end

function alt_sign(m::Rational{Int})
    if m.den == 2
        return (m.num % 4 == 1) ? im : -im
    elseif m.den == 1
        return alt_sign(m.num)
    else
        error("Unsupported rational in alt_sign: $m")
    end
end

@inline x_factorial(x, n, m) = exp(log(x) - logfactorial(m) + logfactorial(n))

function xyz2sph_real(lx::Int, ly::Int, lz::Int, m::Int)
    if isodd(lx + ly + m)
        return 0.0
    end

    abs_m = abs(m)
    l = lx + ly + lz
    j = div(lx + ly - abs_m, 2)

    c0 = sqrt(x_factorial((2l + 1) / (4π), l - abs_m, l + abs_m)) * (0.5^l) / factorial(l)

    cpi = 0.0
    for i in max(0, j):div(l - abs_m, 2)
        cp0 = binomial(l, i) * binomial(i, j) * x_factorial(1, 2l - 2i, l - abs_m - 2i)
        cpi += alt_sign(i) * cp0
    end
    c0 *= cpi

    cpk = 0.0
    for k in max(0, div(lx - abs_m + 1, 2)):min(j, div(lx, 2))
        cp0 = binomial(j, k) * binomial(abs_m, lx - 2k)
        r = (abs_m - lx + 2k) % 4

        if m >= 0
            if r == 0
                cpk += cp0
            elseif r == 2
                cpk -= cp0
            end
        else
            if r == 1
                cpk -= cp0
            elseif r == 3
                cpk += cp0
            end
        end
    end

    if m == 0
        return c0 * cpk
    else
        return sign(m) * sqrt(2.0) * c0 * cpk
    end
end

function cart2sph_hardcoded(l::Int)
    if l == 0
        return Matrix{Float64}(I, 1, 1)
    elseif l == 1
        return Matrix{Float64}(I, 3, 3)
    elseif l == 2
        C = zeros(6, 5)
        C[2, 1] = 1.0
        C[5, 2] = 1.0
        C[1, 3] = -1 / sqrt(3)
        C[4, 3] = -1 / sqrt(3)
        C[6, 3] =  2 / sqrt(3)
        C[3, 4] = 1.0
        C[1, 5] =  1.0
        C[4, 5] = -1.0
        return C
    else
        error("cart2sph_hardcoded only supports l = 0,1,2")
    end
end

function cart2sph_generic(l::Int)
    if l == 0
        return Matrix{Float64}(I, 1, 1)
    elseif l == 1
        return Matrix{Float64}(I, 3, 3)
    else
        vals = Float64[]
        for lx in l:-1:0
            for ly in (l - lx):-1:0
                lz = l - lx - ly
                for m in -l:l
                    push!(vals, xyz2sph_real(lx, ly, lz, m))
                end
            end
        end
        ncart = div((l + 1) * (l + 2), 2)
        nsph  = 2l + 1
        return transpose(reshape(vals, nsph, ncart))
    end
end

function cart2sph(l::Int)
    l <= 2 ? cart2sph_hardcoded(l) : cart2sph_generic(l)
end

function block_diag(matrices::Vector{<:AbstractMatrix})
    rows = sum(size.(matrices, 1))
    cols = sum(size.(matrices, 2))
    T = promote_type(map(eltype, matrices)...)
    B = zeros(T, rows, cols)

    r0 = 1
    c0 = 1
    for M in matrices
        nr, nc = size(M)
        B[r0:r0+nr-1, c0:c0+nc-1] = M
        r0 += nr
        c0 += nc
    end
    return B
end

function cart2sph_coeff(bs::BasisSet)
    ls = basis_angular.(bs.basis)
    max_l = maximum(ls)
    c2s_l = [cart2sph(l) for l in 0:max_l]
    return block_diag([c2s_l[l+1] for l in ls])
end

function compute_spherical_AO_integrals(bs_in::BasisSet)
    bs_cart =
        if (bs_in.basis[1] isa SphericalShell) || (bs_in.lib != :acsint)
            BasisSet(bs_in.name, bs_in.atoms; spherical=false, lib=:acsint)
        else
            bs_in
        end

    C = cart2sph_coeff(bs_cart)

    S_cart_raw = overlap(bs_cart)
    T_cart_raw = kinetic(bs_cart)

    S_sph_raw = C' * S_cart_raw * C
    T_sph_raw = C' * T_cart_raw * C

    d_sph = 1.0 ./ sqrt.(diag(S_sph_raw))
    replace!(d_sph, Inf => 0.0)
    replace!(d_sph, NaN => 0.0)

    S_sph = S_sph_raw .* (d_sph * d_sph')
    T_sph = T_sph_raw .* (d_sph * d_sph')

    N_cart = bs_cart.nbas
    V_cart_raw = zeros(N_cart, N_cart, N_cart)
    EPI_1e3c!(V_cart_raw, bs_cart)

    N_sph = size(C, 2)

    V_tmp1 = reshape(C' * reshape(V_cart_raw, N_cart, :), N_sph, N_cart, N_cart)
    V_p1   = permutedims(V_tmp1, (2, 1, 3))
    V_tmp2 = reshape(C' * reshape(V_p1, N_cart, :), N_sph, N_sph, N_cart)
    V_p2   = permutedims(V_tmp2, (3, 2, 1))
    V_tmp3 = reshape(C' * reshape(V_p2, N_cart, :), N_sph, N_sph, N_sph)

    V_sph_raw = permutedims(V_tmp3, (2, 3, 1))

    V_sph = zeros(N_sph, N_sph, N_sph)
    @inbounds for k in 1:N_sph, j in 1:N_sph, i in 1:N_sph
        V_sph[i, j, k] = V_sph_raw[i, j, k] * d_sph[i] * d_sph[j] * d_sph[k]
    end

    return T_sph, V_sph, S_sph
end

function compute_spherical_AO_integrals_sparse(bs_in::BasisSet; cutoff=1e-12)
    bs_cart =
        if (bs_in.basis[1] isa SphericalShell) || (bs_in.lib != :acsint)
            BasisSet(bs_in.name, bs_in.atoms; spherical=false, lib=:acsint)
        else
            bs_in
        end

    C = cart2sph_coeff(bs_cart)

    S_cart_raw = overlap(bs_cart)
    T_cart_raw = kinetic(bs_cart)

    S_sph_raw = C' * S_cart_raw * C
    T_sph_raw = C' * T_cart_raw * C

    d_sph = 1.0 ./ sqrt.(diag(S_sph_raw))
    replace!(d_sph, Inf => 0.0)
    replace!(d_sph, NaN => 0.0)

    S_sph = S_sph_raw .* (d_sph * d_sph')
    T_sph = T_sph_raw .* (d_sph * d_sph')

    N_cart = bs_cart.nbas
    N_sph  = size(C, 2)

    cart_cutoff = cutoff * 1e-3
    indexes, vals = sparseEPI_1e3c(bs_cart, bs_cart, cart_cutoff)

    V_sph_raw = zeros(N_sph, N_sph, N_sph)
    @inbounds for n in eachindex(vals)
        i, j, k = indexes[n]
        v = vals[n]
        abs(v) <= cart_cutoff && continue

        for c in 1:N_sph
            Ckc = C[k, c]
            abs(Ckc) <= cutoff && continue
            for b in 1:N_sph
                Cjb = C[j, b]
                abs(Cjb) <= cutoff && continue
                Cjb_Ckc = Cjb * Ckc
                for a in 1:N_sph
                    V_sph_raw[a, b, c] += C[i, a] * Cjb_Ckc * v
                end
            end
        end
    end

    out_indices = NTuple{3,Int}[]
    out_vals    = Float64[]
    @inbounds for c in 1:N_sph, b in 1:N_sph, a in 1:N_sph
        v = V_sph_raw[a, b, c] * d_sph[a] * d_sph[b] * d_sph[c]
        if abs(v) > cutoff
            push!(out_indices, (a, b, c))
            push!(out_vals, v)
        end
    end

    return T_sph, (out_indices, out_vals), S_sph
end