# ---------------------------------------------------------------------------
# Multiplet-complete parameterisation of the reduction matrix Wb.
#
# H_out = Wb H_AO Wb'.  A rotation of the structure sends H_AO -> D' H_AO D
# with D block diagonal in the AO (l,m) multiplets, so the eigenvalues of
# H_out are invariant iff the row space of Wb is D-invariant.  The D-invariant
# subspaces are exactly ⊕_l (C_l ⊗ R^{2l+1}): one coefficient per (l, shell),
# shared by all 2l+1 m components of that shell,
#
#     Wb_atom = ⊕_l ( C_l ⊗ I_{2l+1} ),      C_l :: n_out_l × n_shell_l
#
# so NORB = Σ_l n_out_l (2l+1) is quantised in multiplets.  A dense
# NORB × Nb Wb, which is what ACEv3Center initialises by default, is not
# D-invariant: on 2-atom Si / def2-SVP a general rotation moves the Gamma
# eigenvalues by tens of eV, against ~1e-11 eV here.  Rotations from the cubic
# point group are exact either way.
# ---------------------------------------------------------------------------

export wb_multiplet_layout, assemble_Wb_multiplet, wb_multiplet_norb

"""
    wb_multiplet_layout(bs) -> (ls, offsets, Nb, shells_by_l)

`ls[s]` is the l of shell s of atom 1, `offsets[s]` the AO offset of that shell
within the atom, `Nb` the atom's AO count, and `shells_by_l[l+1]` the shell
indices with that l, in order.
"""
function wb_multiplet_layout(bs)
    shells, _ = shells_for_atom(bs, 1)
    ls = [Int(sh.l) for sh in shells]
    sizes = [2l + 1 for l in ls]
    offsets = cumsum([0; sizes[1:end-1]])
    Nb = sum(sizes)
    lmax = maximum(ls)
    shells_by_l = [findall(==(l), ls) for l in 0:lmax]
    return ls, offsets, Nb, shells_by_l
end

"""
    wb_multiplet_norb(bs, n_out) -> Int

`Σ_l n_out[l+1] * (2l+1)`, the NORB a multiplet parameterisation produces.
"""
function wb_multiplet_norb(bs, n_out)
    _, _, _, shells_by_l = wb_multiplet_layout(bs)
    return sum(n_out[l+1] * (2l + 1) for l in 0:(length(shells_by_l)-1)
               if n_out[l+1] > 0)
end

"""
    assemble_Wb_multiplet(C, bs, n_out) -> Matrix

Expand the per-l coefficient blocks `C[l+1] :: n_out[l+1] × n_shell_l` into the
atom's full `NORB × Nb` reduction matrix.  Built row by row with no mutation so
it differentiates.

Row order: for l = 0, 1, ..., for each output a of that l, for each m.  That
keeps whole multiplets contiguous, which is what makes the row space
D-invariant.
"""
function assemble_Wb_multiplet(C, bs, n_out)
    # The shell layout, the (l,a,m) row order and the one-hot columns are
    # structural and carry no derivative.  They go through
    # ignore_derivatives because Zygote cannot differentiate the tuple
    # destructuring of wb_multiplet_layout, and the rows are built without
    # push! because mutation is not differentiable either.
    layout = ChainRulesCore.ignore_derivatives() do
        _, offsets, Nb, shells_by_l = wb_multiplet_layout(bs)
        lmax = length(shells_by_l) - 1
        for l in 0:lmax
            nout = n_out[l+1]
            nout == 0 && continue
            isempty(shells_by_l[l+1]) && error(
                "n_out[$(l+1)] = $nout but the basis has no l = $l shell")
        end
        idx = [(l, a, m) for l in 0:lmax if n_out[l+1] > 0
                         for a in 1:n_out[l+1] for m in 1:(2l + 1)]
        (offsets = offsets, Nb = Nb, shells_by_l = shells_by_l,
         lmax = lmax, idx = idx)
    end

    T = promote_type(eltype.(C)...)
    for l in 0:layout.lmax
        nout = n_out[l+1]
        nout == 0 && continue
        size(C[l+1]) == (nout, length(layout.shells_by_l[l+1])) || error(
            "C[$(l+1)] is $(size(C[l+1])), expected " *
            "$((nout, length(layout.shells_by_l[l+1])))")
    end

    rows = map(layout.idx) do lam
        l, a, m = lam
        shells = layout.shells_by_l[l+1]
        # one coefficient per shell, the SAME for every m of that shell
        sum(C[l+1][a, si] .*
            ChainRulesCore.ignore_derivatives() do
                j0 = layout.offsets[s] + m
                [j == j0 ? one(T) : zero(T) for j in 1:layout.Nb]
            end
            for (si, s) in enumerate(shells))
    end
    return reduce(vcat, (permutedims(r) for r in rows))
end
