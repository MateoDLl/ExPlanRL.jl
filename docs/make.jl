using ExPlanRL
using Documenter

DocMeta.setdocmeta!(ExPlanRL, :DocTestSetup, :(using ExPlanRL); recursive=true)

makedocs(;
    modules=[ExPlanRL],
    authors="Mateo Llivisaca",
    sitename="ExPlanRL.jl",
    format=Documenter.HTML(;
        canonical="https://MateoDLl.github.io/ExPlanRL.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MateoDLl/ExPlanRL.jl",
    devbranch="master",
)
