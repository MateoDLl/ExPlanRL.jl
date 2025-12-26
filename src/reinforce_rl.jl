function idx_to_state(
    centralidades::Dict{String, Dict{Int, Matrix}}, 
    num_candidatos::Int,
    Stage,
    caseStudyData
)::Vector{Vector}

    indexsel = [1, 2, 3]
    indexpw  = [6, 7]
    state_max = Dict{Int, Matrix{Float32}}()

    for etapa in 1:Stage
        mat_tot_etapa = nothing

        for (_, dict_etapas) in centralidades
            mat = dict_etapas[etapa]
            n_node = size(mat, 1)
            mat_red = zeros(Float32, n_node, 3)
            mat_red[:, :] .= mat[:, indexsel]

            # Detectar nodos aislados
            # mat_red_iso = ifelse.(mat_red .== 0.0, 1.0, 0.0)

            # Coeficiente de variación personalizado
            # mat_red = (((((1 / n_node) .- mat_red).^2) ./ n_node).^0.5) ./ (1 / n_node)
            # mat_red = mat_red ./ sum(mat_red, dims = 1)
            # mat_red .+= mat_red_iso

            # Concatenar con info de potencia
            mat_tot = round.(hcat(mat_red, mat[:, indexpw]), digits = 5)

            # Marcar potencia negativa
            #mat_tot[:, 5] .= ifelse.(mat_tot[:, 5] .< 0, 1.0, 0.0)

            # Acumular el máximo por posición
            if mat_tot_etapa === nothing
                mat_tot_etapa = copy(mat_tot)
            else
                mat_tot_etapa = max.(mat_tot_etapa, mat_tot)
            end
        end

        state_max[etapa] = mat_tot_etapa
    end

    # Generar el vector de estado por línea candidata y etapa
    States = Vector{Vector{Float32}}(undef, num_candidatos)
    for i in 1:round(Int, num_candidatos / Stage)
        for etapa in 1:Stage
            f = caseStudyData["v_line_node"][i][1]
            t = caseStudyData["v_line_node"][i][2]
            cap_u  = caseStudyData["Mat_Cap_U"][i, 1]
            cost_u = caseStudyData["Mat_cost_U"][i, 1]
            idx = (etapa - 1) * round(Int, num_candidatos / Stage) + i
            # mean
            mean_idx = (state_max[etapa][f, :] .+ state_max[etapa][t, :])/2
            # diff
            diff_idx = abs.(state_max[etapa][f, :] .- state_max[etapa][t, :])

            States[idx] = vcat(mean_idx, diff_idx, [cap_u, cost_u])
        end
    end

    return States
end

function eval_cty_tnep(data::Dict, top::Matrix{Int})::Tuple{Float32,Bool,Dict}
    FO = 1e4
    feas = true 
    if data["ReactiveCompesation"]
        _,FO,St,_,cty = ACOPF_Extensions.solve_tnep_N1_idx_rc(data,top; subgra=false)
        if !(Int(St) in [1 4 7 10])
            FO = (FO+1)*3
            feas = false
        end 
        return FO,feas,cty
    else
        _,FO,St,_,cty = ACOPF_Extensions.solve_tnep_N1_idx_nrc(data,top; subgra=false)
        if !(Int(St) in [1 4 7 10])
            FO = (FO+1)*3
            feas = false
        end 
        return FO,feas,cty
    end 
end

function eval_ap_tnep(data::Dict, top::Matrix{Int},mejor_FO_batch, factor)::Tuple{Float32,Bool,Float64}
    FO = 1e6
    feas = false
    if data["ReactiveCompesation"]
        _,FO,St,_,ap = ACOPF_Extensions.solve_tnep_N1_rc_AP(data,top; subgra=false)
    else
        _,FO,St,_,ap = ACOPF_Extensions.solve_tnep_N1_nrc_AP(data,top; subgra=false)
    end 
    FO = FO - ap*1e9
    FO = FO + ap*1e2
    if !(Int(St) in [1 4 7 10])
        FO = FO + (1.5+factor)*mejor_FO_batch
    end 
    return FO,feas,ap
end


# -------------------------
# Simulación de entorno
# -------------------------
mutable struct RedElectricaEntorno
    num_candidatos::Int
    estado_actual::Vector{Vector{Float32}}
    acciones::Vector{CartesianIndex{2}}
    topologia::Matrix{Int64}
    mejor_FO::Union{Nothing, Float64}
    factor::Float64
    actual_FO::Union{Nothing, Float64}
    promedio_recompensa::Float64
    κ::Int
    σ::Int
    mejor_FO_batch::Union{Nothing, Float64}
    factor_batch::Float64
end

function RedElectricaEntorno(num_candidatos::Int, Stage::Int, vk, vs, caseStudyData, factor)
    topologia = zeros(Int, num_candidatos, Stage)
    acciones_iniciales = vec(collect(CartesianIndices(topologia)))
    _,_,estado = eval_cty_tnep(caseStudyData, topologia)
    estado_inicial = idx_to_state(estado, num_candidatos*Stage, Stage, caseStudyData)

    return RedElectricaEntorno(num_candidatos*Stage, estado_inicial, acciones_iniciales, topologia, 1e18, factor, nothing,1.0,vk,vs, 1e18, 0.0)
end

function reset!(entorno::RedElectricaEntorno, caseStudyData)
    nc,st = size(entorno.topologia)
    entorno.topologia = zeros(Int, nc, st)
    entorno.acciones = vec(collect(CartesianIndices(entorno.topologia)))
    _,_,estado = eval_cty_tnep(caseStudyData, entorno.topologia)
    entorno.estado_actual = idx_to_state(estado, entorno.num_candidatos, caseStudyData["Stage"], caseStudyData)
    return entorno.estado_actual
end

function acciones_disponibles(matriz::Matrix{Int}, limite::Int)
    posibles = Vector{CartesianIndex{2}}(undef,0)
    for i in 1:size(matriz, 1)
        if sum(matriz[i, :]) < limite
            for j in 1:size(matriz, 2)
                push!(posibles,CartesianIndex{2}(i, j))
            end
        end
    end
    return posibles
end

function line_node(caseStudyData::Dict)::Vector{Tuple{Int, Int}}
    v = Vector{Tuple{Int, Int}}(undef, length(caseStudyData["ne_branch"]))
    for (i,dat) in caseStudyData["ne_branch"]
        v[parse(Int,i)] = (dat["f_bus"],dat["t_bus"])
    end
    return v
end

function evaluar_red!(entorno, FO, feas, n_action)
    entorno.actual_FO = FO

    # Caso: acción inválida
    if !feas && !n_action
        return false, 0.0, false
    end

    # Si no existe aún un mejor FO global (inicio del entrenamiento)
    if isnothing(entorno.mejor_FO_batch)
        entorno.mejor_FO_batch = FO
        entorno.factor_batch = 1
        entorno.factor = 1
        return feas*!n_action, 1.0, true
    end

    # ---------- Recompensa BASE (siempre batch-consistente) ----------
    x = entorno.mejor_FO_batch / FO
    #x = min(x, 1.0)  # clamp por seguridad numérica

    ratio = 1 - exp(-entorno.κ * x^entorno.σ)
    reward = entorno.factor_batch * ratio

    # ---------- Detección de nuevo Best (solo señal, no update) ----------
    new_best = false
    if feas && FO < entorno.mejor_FO_batch
        new_best = true
    end

    return feas*!n_action, reward, new_best
end


function step!(entorno::RedElectricaEntorno, accion::CartesianIndex, caseStudyData, n_action)
    entorno.topologia[accion] += 1
    FO,feas,estado = eval_cty_tnep(caseStudyData, entorno.topologia)
    if !feas && n_action
        FO,_,_ = eval_ap_tnep(caseStudyData, entorno.topologia,entorno.mejor_FO_batch, entorno.factor_batch)
    end
    entorno.estado_actual = idx_to_state(estado, entorno.num_candidatos, caseStudyData["Stage"], caseStudyData)
    terminal, recompensa, new_best = evaluar_red!(entorno,FO,feas, n_action)
    estado_siguiente = copy(entorno.estado_actual)
    return (estado_siguiente, recompensa, terminal, new_best, FO)
end

function seleccionar_accion_policy(policy_model, estado, acciones_disponibles, nlines; k=10, stocas=true)
    q_input = hcat(estado...)

    logits = vec(policy_model(q_input))

    cand_idx = [Int(a[1] + (a[2]-1)*nlines) for a in acciones_disponibles]

    cand_logits = logits[cand_idx]

    if any(isnan, cand_logits) || any(isinf, cand_logits)
        accion_idx = rand(cand_idx)
    else
        k_eff = min(k, length(cand_idx))
        topk_order = partialsortperm(cand_logits, 1:k_eff, rev = true)

        topk_idx = cand_idx[topk_order]
        topk_logits = cand_logits[topk_order]

        if stocas
            w = exp.(topk_logits .- maximum(topk_logits))
            if any(isnan, w) || any(isinf, w) || sum(w) ≤ 0
                accion_idx = rand(topk_idx)
            else
                accion_idx = sample(topk_idx, Weights(w))
            end
        else
            accion_idx = topk_idx[argmax(topk_logits)]
        end
    end

    col = div(accion_idx - 1, nlines) + 1
    row = mod(accion_idx - 1, nlines) + 1

    return CartesianIndex(row, col), accion_idx
end

# function seleccionar_accion_policy(policy_model, estado, acciones_disponibles, nlines;stocas::Bool=true)
#     q_input = hcat(estado...)  # Estado en columnas
#     # probs = NNlib.softmax(vec(policy_model(q_input)))  # Probabilidades por acción
#     # Filtrar solo acciones disponibles y normalizar
#     # mask = zeros(Float32, length(probs))
#     # for a in acciones_disponibles
#     #     idx = Int(a[1] + (a[2] - 1) * nlines)
#     #     mask[idx] = probs[idx]
#     # end
#     # Validación y normalización segura
#     # if any(isnan, mask) || any(isinf, mask) || sum(mask) ≤ 0
#     #     #println("Advertencia: Mask inválido (NaN, Inf o suma 0). Se usará distribución uniforme.")
#     #     for a in acciones_disponibles
#     #         idx = Int(a[1] + (a[2] - 1) * nlines)
#     #         mask[idx] = 1.0
#     #     end
#     # end

#     #mask_norm = mask / sum(mask)

#     logits = vec(policy_model(q_input))
#     mask_logits = fill(-Inf32, length(logits))
#     for a in acciones_disponibles
#         idx = Int(a[1] + (a[2] - 1) * nlines)
#         mask_logits[idx] = logits[idx]
#     end

#     if stocas
#         #accion_idx = sample(1:length(mask_norm), Weights(mask_norm))
#         accion_idx = sample(1:length(mask_logits), Weights(exp.(mask_logits .- maximum(mask_logits))))
#     else
#         #accion_idx = argmax(mask_norm)
#         accion_idx = argmax(mask_logits)
#     end
#     # Convertir indice a CartesianIndex
#     p_en = div(accion_idx - 1, nlines) + 1
#     rest = mod(accion_idx - 1, nlines) + 1
#     return CartesianIndex(rest, p_en), accion_idx
# end

# --- Función para calcular retornos con descuento ---
function calcular_retorno(recompensas, γ)
    R = 0.0
    returns = Float32[]
    for r in reverse(recompensas)
        R = r + γ * R
        push!(returns, R)
    end
    return reverse(returns)
end

function kl_policy(model_new, model_best, estado)
    estado = hcat(estado...)
    logits_new = vec(model_new(estado))
    logits_best = vec(model_best(estado))

    p = NNlib.softmax(logits_new)
    q = NNlib.softmax(logits_best)

    return sum(p .* (log.(p .+ 1e-12) .- log.(q .+ 1e-12)))
end

function kl_batch(model_new, model_best, estados)
    return mean(kl_policy(model_new, model_best, s) for s in estados)
end

# --- Entrenamiento con REINFORCE ---
function entrenar_reinforce_batch_baseline!(num_episodios, entorno, policy_model, opt, batch_size, γ, 
    perdidas_por_batch, recompensas_por_episodios, caseStudyData;
    kl_umbral = 0.02,β_max = 0.6, β_min = 0.01, ajuste_beta = 0.03)
    opt_state = Flux.setup(opt, policy_model)
    nlines =caseStudyData["nlines"]
    buffer_estados = []
    buffer_acciones = []
    buffer_retornos = []
    buffer_ventajas = []

    val_fo = Float64[]
    ps = Flux.trainable(policy_model)
    mejor_trayectoria = ([],[],[],[])

    episodio = 1
    β = 0.2     # inicial
   
    ventana_val = 5
    val_fo = Float32[]
    best_val_fo = Inf
    best_model = deepcopy(policy_model) 
    best_candidates = []
    best_global = 10e29

    while episodio < num_episodios
        estado = reset!(entorno, caseStudyData)
        acciones_disp = entorno.acciones

        estados_epi = Vector{Vector{Vector{Float32}}}()
        acciones_idx_epi = Int[]
        recompensas_epi = Float32[]

        terminado = false
        total_recompensa = 0.0f0
        n_act = 0
        actfin = false
        while !terminado && !actfin      
            n_act += 1
            actfin = n_act >= 50
            accion, accion_idx = seleccionar_accion_policy(policy_model, estado, acciones_disp, nlines)
            estado_siguiente, recompensa, terminado, new_best, of = step!(entorno, accion, caseStudyData, actfin)

            if new_best
                push!(best_candidates, of)
            end

            push!(estados_epi, estado)
            push!(acciones_idx_epi, accion_idx)
            push!(recompensas_epi, recompensa)

            estado = estado_siguiente
            acciones_disp = acciones_disponibles(entorno.topologia, 3)
            total_recompensa += recompensa
        end
        
        push!(recompensas_por_episodios, total_recompensa)
        retornos = calcular_retorno(recompensas_epi, γ)
        baseline = mean(retornos)
        ventajas = retornos .- baseline
        ventajas = (ventajas .- mean(ventajas)) ./ (std(ventajas) + 1e-8)

        append!(buffer_ventajas, ventajas)
        append!(buffer_estados, estados_epi)
        append!(buffer_acciones, acciones_idx_epi)
        append!(buffer_retornos, retornos)

        # --- Función de pérdida REINFORCE ---
        
        if episodio % batch_size == 0
            # Calcula baseline común
            recompensa_promedio = mean(buffer_retornos)
            ventajas = buffer_retornos .- copy(recompensa_promedio)

            loss_fn(policy_model) = begin
                entradas = [hcat(s...) for s in buffer_estados]

                # Forward único por estado
                logits_list = [vec(policy_model(entrada)) for entrada in entradas]

                # Log-probs usando logsoftmax
                log_probs = [
                    Flux.Losses.logsoftmax(logits)[accion_idx]
                    for (logits, accion_idx) in zip(logits_list, buffer_acciones)
                ]

                # Entropía usando softmax
                entropies = [
                    begin
                        p = NNlib.softmax(logits)
                        -sum(p .* log.(p .+ 1e-12))
                    end
                    for logits in logits_list
                ]

                reinforce_loss = -mean(log_probs .* ventajas)
                entropy_bonus  =  mean(entropies)

                return reinforce_loss - β * entropy_bonus
            end

            loss_val, back = Flux.withgradient(loss_fn, policy_model)
            Flux.Optimise.update!(opt_state, ps, back[1])
            push!(perdidas_por_batch, loss_val)
            
            # Validación determinista
            entorno_test = RedElectricaEntorno(nlines, caseStudyData["Stage"], caseStudyData["vk"], caseStudyData["vs"], caseStudyData, 0.0)
            fo, _, _ = evaluar_red_reinforce(policy_model, entorno_test, caseStudyData)
            push!(val_fo, fo)
            #@info("Episodio: $episodio | β: $β | FO val: $fo")
            if length(val_fo) >= ventana_val
                fo_reciente = mean(@view val_fo[end - ventana_val + 1:end])
                if length(val_fo) >= 2 * ventana_val
                    fo_anterior = mean(@view val_fo[end - 2*ventana_val + 1:end - ventana_val]) 
                    mejora_val = fo_anterior - fo_reciente
                    # --- Ajuste dinámico de β ---
                    if mejora_val <= 0 
                        β = min(β + ajuste_beta, β_max)   # subir β hasta β_max
                    else
                        β = max(β - ajuste_beta, β_min)   # bajar β hasta β_min
                    end
                end
             # --- Guardar mejor modelo ---
                if fo_reciente <= best_val_fo
                    best_val_fo = fo_reciente
                    best_model = deepcopy(policy_model)
                end
            end
            if !isempty(best_candidates)
                best_batch = minimum(best_candidates)

                if isnothing(entorno.mejor_FO) || best_batch < entorno.mejor_FO
                    entorno.mejor_FO = best_batch
                    if length(entorno_test.acciones) > 100
                        entorno.factor += 3
                    else
                        entorno.factor += min(length(best_candidates),2)
                    end
                end
            end
            entorno.mejor_FO_batch = entorno.mejor_FO
            entorno.factor_batch   = entorno.factor       
            # Limpieza
            empty!(buffer_estados)
            empty!(buffer_acciones)
            empty!(buffer_retornos)
            empty!(buffer_ventajas)
            empty!(best_candidates)
        end
        episodio += 1
    end
    return best_model
end

# --- Without baseline (but with batch) ---
function entrenar_reinforce_batch!(num_episodios, entorno, policy_model, opt, batch_size, γ, Β, 
    perdidas_por_batch, recompensas_por_episodios, caseStudyData)
    opt_state = Flux.setup(opt, policy_model)
    buffer_estados = []
    buffer_acciones = []
    buffer_retornos = []
    episodios_acumulados = 0
    ps = Flux.trainable(policy_model)
    nlines =caseStudyData["nlines"]

    for episodio in 1:num_episodios
        estado = reset!(entorno, caseStudyData)
        acciones_disp = entorno.acciones

        estados_epi = Vector{Vector{Vector{Float32}}}()
        acciones_idx_epi = Int[]
        recompensas_epi = Float64[]
        terminado = false
        total_recompensa = 0.0
        count = 0
        while !terminado
            count += 1
            acc = count >= 50
            accion, accion_idx = seleccionar_accion_policy(policy_model, estado, acciones_disp, nlines)
            estado_siguiente, recompensa, terminado = step!(entorno, accion, caseStudyData, acc)
            push!(estados_epi, estado)
            push!(acciones_idx_epi, accion_idx)
            push!(recompensas_epi, recompensa)
            estado = estado_siguiente
            acciones_disp = acciones_disponibles(entorno.topologia, 3)
            total_recompensa += recompensa
        end
        push!(recompensas_por_episodios, total_recompensa)
        retornos = calcular_retorno(recompensas_epi, γ)

        append!(buffer_estados, estados_epi)
        append!(buffer_acciones, acciones_idx_epi)
        append!(buffer_retornos, retornos)
        episodios_acumulados += 1

        if episodios_acumulados >= batch_size
            loss_fn(policy_model) = begin
                entradas = [hcat(s...) for s in buffer_estados]

                log_probs = [
                    Flux.Losses.logsoftmax(vec(policy_model(entrada)))[accion_idx]
                    for (entrada, accion_idx) in zip(entradas, buffer_acciones)
                ]

                entropies = [
                    begin
                        p = NNlib.softmax(vec(policy_model(entrada)))
                        -sum(p .* log.(p .+ 1e-8))
                    end
                    for entrada in entradas
                ]

                reinforce_loss = -mean(log_probs .* buffer_retornos)
                entropy_bonus = mean(entropies)
                return reinforce_loss - Β * entropy_bonus
            end
            loss_val, back = Flux.withgradient(loss_fn, policy_model)
            Flux.Optimise.update!(opt_state, ps, back[1])
            push!(perdidas_por_batch, loss_val)
            empty!(buffer_estados)
            empty!(buffer_acciones)
            empty!(buffer_retornos)
            episodios_acumulados = 0
        end
    end
    if episodios_acumulados > 0
        # Entrenamiento final con episodios restantes
        loss_fn_loc(policy_model) = begin
            entradas = [hcat(s...) for s in buffer_estados]

            log_probs = [
                Flux.Losses.logsoftmax(vec(policy_model(entrada)))[accion_idx]
                for (entrada, accion_idx) in zip(entradas, buffer_acciones)
            ]

            entropies = [
                begin
                    p = NNlib.softmax(vec(policy_model(entrada)))
                    -sum(p .* log.(p .+ 1e-8))
                end
                for entrada in entradas
            ]

            reinforce_loss = -mean(log_probs .* buffer_retornos)
            entropy_bonus = mean(entropies)
            return reinforce_loss - Β * entropy_bonus
        end

        loss_val, back = Flux.withgradient(loss_fn_loc, policy_model)
        Flux.Optimise.update!(opt_state, ps, back[1])
        push!(perdidas_por_batch, loss_val)
    end
end

# Sin batch (actualiza cada episodio, sin baseline) ---
function entrenar_reinforce!(num_episodios, entorno, policy_model, opt, batch, γ, Β, 
    perdidas_por_ep, recompensas_por_episodios, caseStudyData)
    opt_state = Flux.setup(opt, policy_model)
    ps = Flux.trainable(policy_model)
    nlines =caseStudyData["nlines"]

    for episodio in 1:num_episodios
        estado = reset!(entorno, caseStudyData)
        acciones_disp = entorno.acciones

        estados_epi = Vector{Vector{Vector{Float32}}}()
        acciones_idx_epi = Int[]
        recompensas_epi = Float64[]
        terminado = false
        total_recompensa = 0.0
        count = 0
        while !terminado
            count += 1
            acc = count >= 100
            accion, accion_idx = seleccionar_accion_policy(policy_model, estado, acciones_disp,nlines)
            estado_siguiente, recompensa, terminado = step!(entorno, accion, caseStudyData, acc)
            push!(estados_epi, estado)
            push!(acciones_idx_epi, accion_idx)
            push!(recompensas_epi, recompensa)
            estado = estado_siguiente
            acciones_disp = acciones_disponibles(entorno.topologia, 3)
            total_recompensa += recompensa
        end
        push!(recompensas_por_episodios, total_recompensa)
        retornos = calcular_retorno(recompensas_epi, γ)

        loss_fn(policy_model) = begin
            entradas = [hcat(s...) for s in estados_epi]

            log_probs = [
                Flux.Losses.logsoftmax(vec(policy_model(entrada)))[accion_idx]
                for (entrada, accion_idx) in zip(entradas, acciones_idx_epi)
            ]

            entropies = [
                begin
                    p = NNlib.softmax(vec(policy_model(entrada)))
                    -sum(p .* log.(p .+ 1e-8))
                end
                for entrada in entradas
            ]

            reinforce_loss = -mean(log_probs .* retornos)
            entropy_bonus = mean(entropies)
            return reinforce_loss - Β * entropy_bonus
        end

        loss_val, back = Flux.withgradient(loss_fn, policy_model)
        Flux.Optimise.update!(opt_state, ps, back[1])
        push!(perdidas_por_ep, loss_val)
    end
end
# Batch con baseline
function entrenar_reinforce_baseline!(num_episodios, entorno, policy_model, opt, batch, γ, Β,
    perdidas_por_episodio, recompensas_por_episodios, caseStudyData)
    opt_state = Flux.setup(opt, policy_model)
    ps = Flux.trainable(policy_model)
    nlines =caseStudyData["nlines"]

    for episodio in 1:num_episodios
        estado = reset!(entorno, caseStudyData)
        acciones_disp = entorno.acciones

        estados_epi = Vector{Vector{Vector{Float32}}}()
        acciones_idx_epi = Int[]
        recompensas_epi = Float64[]
        total_recompensa = 0.0
        terminado = false
        count = 0
        while !terminado
            count += 1
            acc = count >= 100
            accion, accion_idx = seleccionar_accion_policy(policy_model, estado, acciones_disp, nlines)
            estado_siguiente, recompensa, terminado = step!(entorno, accion, caseStudyData, acc)

            push!(estados_epi, estado)
            push!(acciones_idx_epi, accion_idx)
            push!(recompensas_epi, recompensa)

            estado = estado_siguiente
            acciones_disp = acciones_disponibles(entorno.topologia, 3)
            total_recompensa += recompensa
        end
        push!(recompensas_por_episodios, total_recompensa)
        retornos = calcular_retorno(recompensas_epi, γ)
        baseline = mean(retornos)
        ventajas = retornos .- baseline

        entradas = [hcat(s...) for s in estados_epi]

        loss_fn(policy_model) = begin
            log_probs = [
                Flux.Losses.logsoftmax(vec(policy_model(entrada)))[accion_idx]
                for (entrada, accion_idx) in zip(entradas, acciones_idx_epi)
            ]

            entropies = [
                begin
                    p = NNlib.softmax(vec(policy_model(entrada)))
                    -sum(p .* log.(p .+ 1e-8))
                end
                for entrada in entradas
            ]

            reinforce_loss = -mean(log_probs .* ventajas)
            entropy_bonus = mean(entropies)
            return reinforce_loss - Β * entropy_bonus
        end

        loss_val, back = Flux.withgradient(loss_fn, policy_model)
        Flux.Optimise.update!(opt_state, ps, back[1])
        push!(perdidas_por_episodio, loss_val)
    end
end

function evaluar_red_reinforce(policy_model, entorno::RedElectricaEntorno, caseStudyData; stocástico::Bool = false)
    estado = reset!(entorno, caseStudyData)
    acciones_disp = entorno.acciones
    terminado = false
    total_recompensa = 0.0f0
    count = 0
    num_max = 3
    nlines = caseStudyData["nlines"]
    max_act = 50
    acc = false
    entorno.mejor_FO_batch = 1000.0
    while !terminado && !acc 
        count += 1
        acc = count >= max_act
        accion, accion_idx = seleccionar_accion_policy(policy_model, estado, acciones_disp, nlines, stocas=stocástico)

        # Paso en el entorno
        estado_siguiente, recompensa, terminado = step!(entorno, accion, caseStudyData, acc)

        acciones_disp = acciones_disponibles(entorno.topologia, num_max)
        estado = copy(estado_siguiente)
        total_recompensa += recompensa
        if length(acciones_disp) == 0
            acc = true
        end
        if accion in acciones_disp
            deleteat!(acciones_disp, findfirst(==(accion), acciones_disp))
        end
        if length(acciones_disp)/size(entorno.topologia,2) > 100
            num_max = 2
        else
            num_max = 3
        end
        #println("  FO: $(accion)")
    end
    #@info("Costo: $(entorno.actual_FO)") 
    return entorno.actual_FO, entorno.topologia, terminado
end