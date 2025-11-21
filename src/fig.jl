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
    Plots.savefig(figurevar, path*"/var_"*strVFO*"_.png")

    figurerew = Plots.plot(recompensas, 
        xlabel = "Episode",
        ylabel = "Reward",
        title = LaTeXStrings.LaTeXString("Trainig\\\\  Net $id \\\\"),
        legend = false,
        #seriestype = :bar,
        color = :green,
        grid = false,
    )
    Plots.savefig(figurerew, path*"/rew_"*strVFO*"_.png")
end

function moving_average(data, window)
    return [mean(data[max(1, i - window + 1):i]) for i in 1:length(data)]
end

function plot_with_tendency(perdidas, recompensas, VFO, id, path)
    strVFO = string(id)*"_"*string(round(Int,VFO[1]))
    # Parámetro de suavizado
    window_size = 5  # 
    # --- Gráfico de pérdidas ---
    ma_perdidas = moving_average(perdidas, window_size)
    h1 = Plots.plot(
        perdidas,
        xlabel = "Batch",
        ylabel = "Loss",
        title = LaTeXStrings.LaTeXString("Network $i"),
        legend = :bottomleft,
        label = "Loss",
        color = :crimson,
        grid = false,
    )
    Plots.plot!(h1, ma_perdidas, label = "Trend line", color = :black, lw = 2, ls = :dash)
    Plots.savefig(h1, path*"/var_"*strVFO*"_.png")
    
    # --- Gráfico de recompensas ---
    ma_recompensas = moving_average(recompensas, window_size)
    h2 = Plots.plot(
        recompensas,
        xlabel = "Episode",
        ylabel = "Reward",
        title = LaTeXStrings.LaTeXString("Network $i"),
        legend = :topleft,
        label = "Reward",
        color = :springgreen3,
        grid = false,
    )
    Plots.plot!(h2, ma_recompensas, label = "Trend line", color = :black, lw = 2, ls = :dash)
    Plots.savefig(h2, path*"/rew_"*strVFO*"_.png")
end