using RealMLTB
using Documenter

DocMeta.setdocmeta!(RealMLTB, :DocTestSetup, :(using RealMLTB); recursive=true)

makedocs(;
    modules=[RealMLTB],
    authors="Christoph Ortner <christophortner0@gmail.com>",
    sitename="RealMLTB.jl",
    format=Documenter.HTML(;
        canonical="https://ACEsuit.github.io/RealMLTB.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/RealMLTB.jl",
    devbranch="main",
)
