using RealMLTB
using Molecules, GaussianBasis
using AtomsBuilder, NeighbourLists
using EquivariantTensors
using Unitful: ustrip, Å
using Unitful
using LuxCore
using Random

include("data/generate_data.jl")
include("data/plot_bands.jl")

# -----------------------------
# 0) data + geometry
# -----------------------------
rcut  = 12.5u"Å"
sys   = bulk(:Si)
nlist = PairList(sys, rcut)

X     = position(sys, :)
X_ang = [ustrip.(Å, r) for r in X]
mol   = Molecules.Molecule([
    Molecules.Atom(14, 0.0, X_ang[1]),
    Molecules.Atom(14, 0.0, X_ang[2]),
])

NORB       = 2
kgrid      = [6, 7, 8]
keep_bands = 4

dataset = generate_band_data(sys, kgrid; keep_bands=keep_bands)
klist   = getfield.(dataset, 1)
λ_ref   = getfield.(dataset, 2)

# -----------------------------
# 1) build graphs: cart + sph
# -----------------------------
include("integral/integral.jl")
include("utils/build_graph.jl")

bs_cart = BasisSet("def2-SVP",  mol.atoms; spherical=false, lib=:acsint);
bs_sph  = BasisSet("def2-TZVP", mol.atoms; spherical=true,  lib=:acsint);

G_cart = RealMLTB.nlist2graph(bs_cart, nlist, sys, klist);
G_sph  = nlist2graph(bs_sph,  nlist, sys, klist);

# -----------------------------
# 2) helper: train + eval + plot  (single-level)
# -----------------------------
function fit_eval_plot(name::String, model, ps, st, G, sys, kgrid, keep_bands, λ_ref;
                       iterations::Int=3000)
    println("\n==============================")
    println(">> Training: $name")
    println("==============================")

    ps_opt, xhist = train(G, model, ps, st, keep_bands, λ_ref; iterations=iterations)
    err = RMSE(G, λ_ref, model, ps_opt, st; keep_bands=keep_bands)

    println(">> $name RMSE = $err")

    compute_and_save_bandstructures(G, sys, model, ps_opt, st;
        name=name, kgrid=kgrid, keep_bands=keep_bands
    )

    return ps_opt, st, err, xhist
end

# -----------------------------
# 3) run all models
# -----------------------------
rng = MersenneTwister(1234)
results = Dict{String, Any}()

# --- (1) simple3c (cart) ---
model_simple3c = simple_3c_model(bs_cart, NORB);
ps_simple3c, st_simple3c = LuxCore.setup(rng, model_simple3c);

ps_simple3c, st_simple3c, err_simple3c, hist_simple3c =
    fit_eval_plot("simple3c", model_simple3c, ps_simple3c, st_simple3c,
                  G_cart, sys, kgrid, keep_bands, λ_ref; iterations=3000)

results["simple3c"] = (err=err_simple3c, ps=ps_simple3c, st=st_simple3c)

# --- common multilevel args for ACE models ---
target_Dtot = 5
target_maxl = 5
target_ORD  = 3
base_iters  = 3000
seed        = 1234

# --- (2) acev3c (sph) ---
ps_acev, st_acev, model_acev =
    multilevel_train(G_sph, λ_ref, bs_sph, NORB, rcut;
        model_type      = :acev_3c,
        target_Dtot     = target_Dtot,
        target_maxl     = target_maxl,
        target_ORD      = target_ORD,
        keep_bands      = keep_bands,
        base_iterations = base_iters,
        seed            = seed,
        verbose         = true
    )

err_acev = RMSE(G_sph, λ_ref, model_acev, ps_acev, st_acev; keep_bands=keep_bands)
println("\n>> acev3c RMSE = $err_acev")
compute_and_save_bandstructures(G_sph, sys, model_acev, ps_acev, st_acev;
    name="acev3c", kgrid=kgrid, keep_bands=keep_bands
)
results["acev3c"] = (err=err_acev, ps=ps_acev, st=st_acev)

# --- (3) ace3c_A (sph, Wpsi CaseA) ---
# nψ controls the co-rotating orbital content; must satisfy NORB = Σ_l nψ[l]*(2l+1)
# Example for NORB=2: only s-like -> nψ=(2,0,0,0,0)  (2*1 = 2)
nψ = (2, 0, 0, 0, 0)

ps_aceA, st_aceA, model_aceA =
    multilevel_train(G_sph, λ_ref, bs_sph, NORB, rcut;
        model_type      = :ace3c_A,
        nψ              = nψ,
        target_Dtot     = target_Dtot,
        target_maxl     = target_maxl,
        target_ORD      = target_ORD,
        keep_bands      = keep_bands,
        base_iterations = base_iters,
        seed            = seed,
        verbose         = true
    )

err_aceA = RMSE(G_sph, λ_ref, model_aceA, ps_aceA, st_aceA; keep_bands=keep_bands)
println("\n>> ace3c_A RMSE = $err_aceA")
compute_and_save_bandstructures(G_sph, sys, model_aceA, ps_aceA, st_aceA;
    name="ace3c_A", kgrid=kgrid, keep_bands=keep_bands
)
results["ace3c_A"] = (err=err_aceA, ps=ps_aceA, st=st_aceA)

# --- (4) ace3c_B (sph, Wpsi CaseB) ---
ps_aceB, st_aceB, model_aceB =
    multilevel_train(G_sph, λ_ref, bs_sph, NORB, rcut;
        model_type      = :ace3c_B,   # <-- Case B
        target_Dtot     = target_Dtot,
        target_maxl     = target_maxl,
        target_ORD      = target_ORD,
        keep_bands      = keep_bands,
        base_iterations = base_iters,
        seed            = seed,
        verbose         = true
    )

err_aceB = RMSE(G_sph, λ_ref, model_aceB, ps_aceB, st_aceB; keep_bands=keep_bands)
println("\n>> ace3c_B RMSE = $err_aceB")
compute_and_save_bandstructures(G_sph, sys, model_aceB, ps_aceB, st_aceB;
    name="ace3c_B", kgrid=kgrid, keep_bands=keep_bands
)
results["ace3c_B"] = (err=err_aceB, ps=ps_aceB, st=st_aceB)

# -----------------------------
# 4) summary
# -----------------------------
println("\n==============================")
println("Summary (RMSE, keep_bands=$keep_bands)")
println("==============================")
for key in sort!(collect(keys(results)))
    println(rpad(key, 10), "  ", results[key].err)
end
