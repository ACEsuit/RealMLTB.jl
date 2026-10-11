using RealMLTB
using GaussianBasis
using Molecules
using Test
using LinearAlgebra
import GaussianBasis: num_basis, Acsint

# ---------------------------------------------------------------------------
# generate_V_triple! computes the three-centre overlap  ∫ φi φj φk  with a
# McMurchie-Davidson style Hermite recursion.  Validate it against an exact
# reference built from 1D Gaussian moments -- a completely separate code
# path: the triple product factorises per Cartesian direction, and each 1D
# factor is a polynomial times exp(-q(x-Q)^2), integrable in closed form.
# ---------------------------------------------------------------------------

const A2B = Acsint.ang2bohr

# Cartesian components in the order generate_V_triple! writes them
_cartcomp(l) = [(lx, l - lx - nz, nz) for lx in l:-1:0 for nz in 0:(l - lx)]

_dfac(n) = n <= 0 ? 1.0 : prod(1.0:2.0:n)
_mom1(n, q) = isodd(n) ? 0.0 : _dfac(n - 1) / (2q)^(n ÷ 2) * sqrt(pi / q)

function _int1d(i, j, k, A, B, C, a, b, c)
    q = a + b + c
    Q = (a * A + b * B + c * C) / q
    Kx = a * A^2 + b * B^2 + c * C^2 - q * Q^2
    s = 0.0
    for p in 0:i, r in 0:j, t in 0:k
        s += binomial(i, p) * binomial(j, r) * binomial(k, t) *
             (Q - A)^(i - p) * (Q - B)^(j - r) * (Q - C)^(k - t) *
             _mom1(p + r + t, q)
    end
    return exp(-Kx) * s
end

_raw3c(c1, c2, c3, R1, R2, R3, a, b, c) =
    prod(_int1d(c1[d], c2[d], c3[d], R1[d], R2[d], R3[d], a, b, c) for d in 1:3)

_raw_self(cc, a) =
    prod(_dfac(2cc[d] - 1) / (4a)^cc[d] * sqrt(pi / (2a)) for d in 1:3)

function _v3c_relerr(ls, exps, centres)
    atoms = [Molecules.Atom(1, 0.0, collect(centres[i])) for i in eachindex(ls)]
    shells = [GaussianBasis.CartesianShell(ls[i], [1.0], [exps[i]], atoms[i])
              for i in eachindex(ls)]
    bs = BasisSet("custom", atoms, shells; lib = :acsint)

    S = overlap(bs)

    # the per-function scale GaussianBasis actually applies
    off = 0
    scales = Float64[]
    for sh in bs.basis, cc in _cartcomp(sh.l)
        off += 1
        push!(scales, sqrt(S[off, off] / _raw_self(cc, sh.exp[1])))
    end

    maxerr = 0.0
    maxabs = 0.0
    for i in eachindex(bs.basis), j in eachindex(bs.basis), k in eachindex(bs.basis)
        B1, B2, B3 = bs.basis[i], bs.basis[j], bs.basis[k]
        buf = zeros(num_basis(B1) * num_basis(B2) * num_basis(B3))
        RealMLTB.generate_V_triple!(buf, B1, B2, B3)

        R1 = B1.atom.xyz .* A2B
        R2 = B2.atom.xyz .* A2B
        R3 = B3.atom.xyz .* A2B

        o1 = sum(num_basis(bs.basis[t]) for t in 1:(i - 1); init = 0)
        o2 = sum(num_basis(bs.basis[t]) for t in 1:(j - 1); init = 0)
        o3 = sum(num_basis(bs.basis[t]) for t in 1:(k - 1); init = 0)

        idx = 1
        for (n3, c3) in enumerate(_cartcomp(B3.l)),
            (n2, c2) in enumerate(_cartcomp(B2.l)),
            (n1, c1) in enumerate(_cartcomp(B1.l))

            ref = scales[o1 + n1] * scales[o2 + n2] * scales[o3 + n3] *
                  _raw3c(c1, c2, c3, R1, R2, R3, B1.exp[1], B2.exp[1], B3.exp[1])
            maxerr = max(maxerr, abs(buf[idx] - ref))
            maxabs = max(maxabs, abs(ref))
            idx += 1
        end
    end
    return maxerr / maxabs
end

@testset "generate_V_triple! against exact 1D moments" begin
    ctr = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.85), (0.62, 0.31, 0.21)]
    for (ls, exps) in (([0, 0, 0], [0.8, 1.3, 0.55]),
                       ([1, 0, 1], [0.8, 1.3, 0.55]),
                       ([2, 1, 0], [0.8, 1.3, 0.55]),
                       ([2, 2, 2], [0.8, 1.3, 0.55]),
                       ([3, 2, 1], [0.8, 1.3, 0.55]),
                       ([3, 3, 3], [0.7, 0.9, 1.1]))
        @test _v3c_relerr(ls, exps, ctr) < 1e-12
    end
end

@testset "spherical AO integrals are normalised and symmetric" begin
    atoms = [Molecules.Atom(14, 0.0, [0.0, 0.0, 0.0]),
             Molecules.Atom(14, 0.0, [2.35, 2.35, 2.35])]
    bs = BasisSet("def2-TZVP", atoms; spherical = true, lib = :acsint)

    T, V, S = RealMLTB.compute_spherical_AO_integrals(bs)

    @test maximum(abs, diag(S) .- 1) < 1e-12
    @test maximum(abs, S - S') < 1e-12
    @test maximum(abs, T - T') < 1e-12
    # ∫ φi φj φk is symmetric under every permutation of its three indices
    @test maximum(abs, V - permutedims(V, (2, 1, 3))) < 1e-12
    @test maximum(abs, V - permutedims(V, (3, 2, 1))) < 1e-12
    @test maximum(abs, V - permutedims(V, (1, 3, 2))) < 1e-12
end
