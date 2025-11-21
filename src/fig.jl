function plot_perd_reward(perdidas, recompensas, VFO, id, path)
    
    strVFO = string(id)*"_"*string(round(Int,VFO[1]))
    figurevar = Plots.plot(perdidas, 
        xlabel = "Batch",
        ylabel = "Losses",
        title = LaTeXStrings.LaTeXString("Trainig\\\\  Net $id \\\\"),
        legend = false,
        #seriestype = :bar,
        color = :red,
        grid = false,
    )
    Plots.savefig(figurevar, path*"_var_"*strVFO*"_.png")

    figurerew = Plots.plot(recompensas, 
        xlabel = "Episode",
        ylabel = "Reward",
        title = LaTeXStrings.LaTeXString("Trainig\\\\  Net $id \\\\"),
        legend = false,
        #seriestype = :bar,
        color = :green,
        grid = false,
    )
    Plots.savefig(figurerew, path*"_rew_"*strVFO*"_.png")
end