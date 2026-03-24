using FundamentalFrequencies
using Documenter

DocMeta.setdocmeta!(FundamentalFrequencies, :DocTestSetup, :(using FundamentalFrequencies); recursive=true)

makedocs(;
    modules=[FundamentalFrequencies],
    authors="mattsignorelli <mgs255@cornell.edu> and contributors",
    sitename="FundamentalFrequencies.jl",
    format=Documenter.HTML(;
        canonical="https://bmad-sim.github.io/FundamentalFrequencies.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/bmad-sim/FundamentalFrequencies.jl",
    devbranch="main",
)
