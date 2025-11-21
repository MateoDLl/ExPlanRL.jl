function plot_perd_reward(perdidas, recompensas, VFO, id, path)
    
    strVFO = string(id)*"_"*string(round(Int,VFO[1]))
    figurevar = plot(perdidas, 
        xlabel = "Batch",
        ylabel = "Losses",
        title = LaTeXStrings.LaTeXString("Trainig\\\\  Net $id \\\\"),
        legend = false,
        #seriestype = :bar,
        color = :red,
        grid = false,
    )
    savefig(figurevar, path*"_var_"*strVFO*"_.png")

    figurerew = plot(recompensas, 
        xlabel = "Episode",
        ylabel = "Reward",
        title = LaTeXStrings.LaTeXString("Trainig\\\\  Net $id \\\\"),
        legend = false,
        #seriestype = :bar,
        color = :green,
        grid = false,
    )
    savefig(figurerew, path*"_rew_"*strVFO*"_.png")
end