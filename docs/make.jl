using NAFF
using Documenter

DocMeta.setdocmeta!(NAFF, :DocTestSetup, :(using NAFF); recursive=true)

makedocs(;
    modules=[NAFF],
    authors="mattsignorelli <mgs255@cornell.edu> and contributors",
    sitename="NAFF.jl",
    format=Documenter.HTML(;
        canonical="https://mattsignorelli.github.io/NAFF.jl",
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
