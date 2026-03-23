using QuasiperiodicFrequencies
using Documenter

DocMeta.setdocmeta!(QuasiperiodicFrequencies, :DocTestSetup, :(using QuasiperiodicFrequencies); recursive=true)

makedocs(;
    modules=[QuasiperiodicFrequencies],
    authors="mattsignorelli <mgs255@cornell.edu> and contributors",
    sitename="QuasiperiodicFrequencies.jl",
    format=Documenter.HTML(;
        canonical="https://mattsignorelli.github.io/QuasiperiodicFrequencies.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mattsignorelli/NAFF.jl",
    devbranch="main",
)
