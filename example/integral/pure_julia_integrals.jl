module PureJuliaIntegrals

using LinearAlgebra
using SpecialFunctions
using GaussianBasis
using Molecules
using RealMLTB
import GaussianBasis: Acsint

const ang2bohr = Acsint.ang2bohr

# =========================
# Cartesian -> Spherical
# =========================

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

"""
Generic real cart->sph coefficient for one Cartesian monomial x^lx y^ly z^lz
to real spherical component m.

This is kept as a generic fallback for l >= 3.
"""
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

"""
PySCF-compatible convention for l = 0, 1, 2.

Matrix shape is ncart × nsph.
Cartesian ordering:
- l=0: [s]
- l=1: [x, y, z]
- l=2: [xx, xy, xz, yy, yz, zz]

Spherical ordering:
- l=0: [s]
- l=1: [px, py, pz]
- l=2: [dxy, dyz, dz2, dxz, dx2-y2]
"""
function cart2sph_hardcoded(l::Int)
    if l == 0
        return Matrix{Float64}(I, 1, 1)

    elseif l == 1
        return Matrix{Float64}(I, 3, 3)

    elseif l == 2
        C = zeros(6, 5)

        # dxy
        C[2, 1] = 1.0

        # dyz
        C[5, 2] = 1.0

        # dz^2 = (2zz - xx - yy) / sqrt(3)
        C[1, 3] = -1 / sqrt(3)
        C[4, 3] = -1 / sqrt(3)
        C[6, 3] =  2 / sqrt(3)

        # dxz
        C[3, 4] = 1.0

        # dx2-y2 = xx - yy
        C[1, 5] =  1.0
        C[4, 5] = -1.0

        return C
    else
        error("cart2sph_hardcoded only supports l = 0,1,2")
    end
end

"""
Generic cart->sph matrix, using the original analytic formula.

For l >= 3 this gives a mathematically valid real spherical transform, but the
exact ordering/sign convention has not yet been exhaustively matched against PySCF.
"""
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

"""
Hybrid version:
- l <= 2: hardcoded, PySCF-compatible for the currently validated shells
- l >= 3: generic fallback
"""
function cart2sph(l::Int)
    if l <= 2
        return cart2sph_hardcoded(l)
    else
        return cart2sph_generic(l)
    end
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

# =========================
# Pure Julia spherical AO integrals
# =========================

"""
    compute_spherical_AO_integrals(bs_in::BasisSet)

Pure Julia implementation:
1. Rebuild a Cartesian basis if needed
2. Compute raw Cartesian S/T/V
3. Transform to spherical basis using internal Julia cart2sph convention
4. Normalize spherical basis so that diag(S)=1
"""
function compute_spherical_AO_integrals(bs_in::BasisSet)
    bs_cart =
        if (bs_in.basis[1] isa SphericalShell) || (bs_in.lib != :acsint)
            BasisSet(bs_in.name, bs_in.atoms; spherical=false, lib=:acsint)
        else
            bs_in
        end

    C = cart2sph_coeff(bs_cart)

    # 2-center
    S_cart_raw = overlap(bs_cart)
    T_cart_raw = kinetic(bs_cart)

    S_sph_raw = C' * S_cart_raw * C
    T_sph_raw = C' * T_cart_raw * C

    d_sph = 1.0 ./ sqrt.(diag(S_sph_raw))
    replace!(d_sph, Inf => 0.0)
    replace!(d_sph, NaN => 0.0)

    S_sph = S_sph_raw .* (d_sph * d_sph')
    T_sph = T_sph_raw .* (d_sph * d_sph')

    # 3-center
    N_cart = bs_cart.nbas
    V_cart_raw = zeros(N_cart, N_cart, N_cart)
    RealMLTB.EPI_1e3c!(V_cart_raw, bs_cart)

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

export cart2sph,
       cart2sph_generic,
       cart2sph_hardcoded,
       cart2sph_coeff,
       compute_spherical_AO_integrals

end # module