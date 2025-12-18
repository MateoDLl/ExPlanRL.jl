function evaluar_parametros(params, semilla, caseData, timeGlobal, kl_um ,βmax , βmin, a_beta,
    hidden1, hidden2, network; policy_model = nothing)

    nlines = caseData["nlines"]
    Stage = caseData["Stage"]
    caseData["v_line_node"] = line_node(caseData)
    vk = round(Int,params[1])
    vs = round(Int,params[2])
    caseData["vk"] = vk
    caseData["vs"] = vs
    frecuencia = round(Int,params[3])
    tasa_aprendizaje = params[4]
    γ = params[5]      # factor de descuento
    nepi = round(Int,params[6])
    # Ejecuta el algoritmo
    if isnothing(policy_model)
        policy_model_0 = Flux.Chain(
            Flux.Dense(12, hidden1, NNlib.relu),
            Flux.Dense(hidden1, hidden2, NNlib.relu),
            Flux.Dense(hidden2, 1)  # un Q-valor por candidato
        )
    else
        policy_model_0 = deepcopy(policy_model)
    end
    perdidas_por_batch  = Float64[]
    recompensas_episodios = Float64[]
    opt = Flux.Adam(tasa_aprendizaje)
    #opt_state = Flux.setup(opt, policy_model)
    entorno = RedElectricaEntorno(nlines, Stage, vk, vs, caseData)  # candidatos
    timeTrain = @elapsed policy_model = entrenar_reinforce_batch_baseline!(nepi, entorno, policy_model_0, opt, frecuencia, γ, perdidas_por_batch, recompensas_episodios,caseData,
                                    kl_umbral = kl_um,β_max = βmax, β_min = βmin, ajuste_beta = a_beta)
    entorno = RedElectricaEntorno(nlines, Stage, vk, vs, caseData)
    
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")

    folder = get_experiment_folder(timeGlobal, base="test")

    filename = joinpath(folder,
                        "resultados_$(nepi)_$(timestamp)_$(network).bson")

    open(joinpath(folder, "PruebasREINFORCE.txt"), "a") do io
        println(io, basename(filename))
    end


    #@save filename timeTrain params nepi perdidas_por_batch
    VFO = evaluar_red_reinforce(policy_model, entorno, caseData)  
    @save filename policy_model timeTrain params nepi perdidas_por_batch VFO semilla recompensas_episodios network  
    # Devuelve el valor a minimizar 
    #return VFO
end

#Random.seed!(1234)
function run_rl_reinforce_train(system::String, rc::Bool, n1::Bool;
    kl_um = 0.02,βmax = 0.6, βmin = 0.01, a_beta = 0.03,
    hidden1=144, hidden2=24, 
    parameters = [[4],[4],[3 6 9],[0.005 0.01], [0.99 0.999], [500]])
    @info("$(system),  N_ep: $(parameters[6])  ") 
    caseStudyData = prepare_case(system, rc, n1)
    p1,p2,p3,p4,p5,p6 = parameters
    correr_experimentos_pmap(p1,p2,p3,p4,p5,p6, caseStudyData, kl_um,βmax, βmin, a_beta,
                            hidden1, hidden2)
end

function run_rl_reinforce_train(system::String, rc::Bool, n1::Bool, path::String;
    kl_um = 0.02,βmax = 0.6, βmin = 0.01, a_beta = 0.03,
    hidden1=144, hidden2=24,
    episodes = 100, best_NN = nothing)
    caseStudyData = prepare_case(system, rc, n1)
    correr_experimentos_trained_pmap(path, caseStudyData, kl_um,βmax, βmin, a_beta,
                                    hidden1, hidden2, episodes, best_NN)
end


function wrapper(parametros_test, semilla, caseStudyData, timeGlobal, kl_um,βmax, βmin, a_beta,
    hidden1, hidden2, net; policy=nothing)
    Random.seed!(semilla)
    evaluar_parametros(parametros_test, semilla, caseStudyData, timeGlobal, kl_um,βmax, βmin, a_beta,
    hidden1, hidden2, net, policy_model = policy)
end


function correr_experimentos_pmap(p1,p2,p3,p4,p5,p6, caseStudyData, kl_um,βmax, βmin, a_beta,
    hidden1, hidden2)
    seed = 1000
    trabajos = []
    timeGlobal = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    for (i, params) in enumerate(Iterators.product(p1, p2, p3, p4, p5, p6))
        semilla = seed + i * 10
        push!(trabajos, (params, semilla, i))
    end
    Distributed.pmap(trabajos) do (parametros_test, semilla, net) 
        wrapper(parametros_test, semilla, caseStudyData, timeGlobal, kl_um,βmax, βmin, a_beta,
                hidden1, hidden2, net)
    end
    
end 

function correr_experimentos_trained_pmap(path_archivo, caseStudyData, kl_um,βmax, βmin, a_beta,
    hidden1, hidden2, episodes, best_NN)
    trabajos = []
    timeGlobal = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")

    folder = dirname(path_archivo)

    open(path_archivo, "r") do archivo

        filenames = collect(eachline(archivo))
        models = Vector{NamedTuple}(undef, length(filenames))

        for (i, filename) in enumerate(filenames)
            @load joinpath(folder, filename) policy_model params semilla network
            models[i] = (
                policy = policy_model,
                params  = params,
                semilla = semilla,
                network = network
            )
        end
        best_policy =
        if isnothing(best_NN)
            nothing
        else
            idx = findfirst(m -> m.network == best_NN, models)
            isnothing(idx) && error("best_NN not found in loaded models")
            deepcopy(models[idx].policy)
        end
        for m in models
            p1, p2, p3, p4, p5, _ = m.params
            new_param = (p1, p2, p3, p4, p5, episodes)

            policy_to_use = isnothing(best_policy) ? m.policy : best_policy

            push!(trabajos, (new_param, m.semilla, policy_to_use, m.network))
        end
    end
    Distributed.pmap(trabajos) do (parametros_test, semilla, policy_model, network)
        wrapper(parametros_test, semilla, caseStudyData, timeGlobal, kl_um,βmax, βmin, a_beta,
        hidden1, hidden2, network, policy = policy_model)
    end
end  

function evaluar_parametros(params, policy_model, nlines, Stage, caseStudyData; sel=false)
    vk = round(Int,params[1])
    vs = round(Int,params[2])
    
    entorno = RedElectricaEntorno(nlines, Stage, vk, vs, caseStudyData)
    VFO, top = evaluar_red_reinforce(policy_model, entorno, caseStudyData, stocástico = sel)  
    return VFO, top
end


function cargar_modelos(path_archivo, vec_idd)
    vec_names = String[]
    vec_results = []
    vec_id = []
    folder = dirname(path_archivo)

    open(path_archivo, "r") do archivo
        for (id, filename) in enumerate(eachline(archivo))
            @load joinpath(folder, filename) policy_model timeTrain params nepi perdidas_por_batch VFO semilla recompensas_episodios network
            push!(vec_names, filename)
            push!(vec_results, (policy_model, params, perdidas_por_batch, recompensas_episodios, VFO))
            push!(vec_id, network)
            plot_with_tendency(perdidas_por_batch, recompensas_episodios, VFO, network, folder) 
        end
    end

    return vec_names, vec_results, vec_id
end

function evaluar_sistemas(vec_results, sistemas, react_comps, contingens;
                            stage::Int = 1, grate::Float64 = 20.0,
                                  drate::Float64 = 10.0, yearst::Int = 1, sel = false)
    vector_total = []

    for (i, sis_train) in enumerate(vec_results)
        vectorRes = []

        for (v1, v2, v3) in Iterators.product(sistemas, react_comps, contingens)
            #println("Sistema:$v1, RC:$v2, Ctg:$v3")
            caseStudyData = prepare_case(v1, v2, v3, 
                                        stage=stage, grate=grate, drate=drate, yearst= yearst)
            nlines = caseStudyData["nlines"]
            caseStudyData["v_line_node"] = line_node(caseStudyData)

            time_test = @elapsed valor, top = evaluar_parametros(
                sis_train[2],       # params
                sis_train[1],       # policy_model
                nlines,
                stage,
                caseStudyData,
                sel = sel
            )

            valido = true
            if isnothing(valor)
                valor = sum(caseStudyData["Mat_cost"] .* top)
                valido = false
            end

            push!(vectorRes, (v1, v2, v3, round(valor, digits=2),
                              round(time_test, digits=2), top, valido))
        end

        push!(vector_total, vectorRes)
    end

    return vector_total
end

# Para que las funciones y variables necesarias estén disponibles en todos los workers
function evaluar_sistemas_worker(sis_train, id, sistemas, react_comps, contingens;
                                             stage::Int=1, grate::Float64=20.0,
                                             drate::Float64=10.0, yearst::Int=1, sel = false)
    vectorRes = []

    for (v1, v2, v3) in Iterators.product(sistemas, react_comps, contingens)
        println("Sistema:$v1, RC:$v2, Ctg:$v3")
        caseStudyData = prepare_case(v1, v2, v3, 
                                     stage=stage, grate=grate, drate=drate, yearst=yearst)
        nlines = caseStudyData["nlines"]
        caseStudyData["v_line_node"] = line_node(caseStudyData)

        time_test = @elapsed valor, top = evaluar_parametros(
            sis_train[2],  # params
            sis_train[1],  # policy_model
            nlines,
            stage,
            caseStudyData,
            sel = sel
        )

        valido = true
        if isnothing(valor)
            valor = sum(caseStudyData["Mat_cost"] .* top)
            valido = false
        end

        push!(vectorRes, (v1, v2, v3, round(valor, digits=2),
                          round(time_test, digits=2), top, valido, id, sis_train[2]))
    end

    return vectorRes
end



"""
    run_reinforce_evaluation(path_pruebas, path_salida,
                             sistemas = ["garverQ", "case24IEEE_P_Glo_r", "case118_cost_R_D_r"],
                             react_comps = [false, true],
                             contingens = [false];
                             vec_id = nothing,
                             stage::Int = 1, grate::Float64 = 20.0,
                             drate::Float64 = 10.0, yearst::Int = 1, select = false)

Ejecuta la evaluación REINFORCE sobre múltiples sistemas eléctricos.
- Devuelve `vector_total` y también guarda los resultados en `path_salida`.
"""
function run_evaluation(path_pruebas::String,
                                  path_salida::String,
                                  sistemas,
                                  react_comps,
                                  contingens;
                                  vec_id = nothing, 
                                  stage::Int = 1, grate::Float64 = 20.0,
                                  drate::Float64 = 10.0, yearst::Int = 1, select = false)

    println("=== Cargando modelos REINFORCE ===")
    # Si no se pasa vec_id → usar todos los índices del archivo
    if isnothing(vec_id)
        total = countlines(path_pruebas)   # Número total de líneas en el archivo
        vec_id = collect(1:total)
        println("No se especificó vec_id → usando todos los modelos (1:$total).")
    else
        println("Usando vec_id = $vec_id")
    end

    # Cargar los modelos entrenados
    vec_names, vec_results, vec_net = cargar_modelos(path_pruebas, vec_id)

    println("=== Ejecutando evaluación sobre sistemas ===")

    # Usando pmap
    vector_total = Distributed.pmap((sis_train, id) -> evaluar_sistemas_worker(sis_train, id, sistemas, react_comps, contingens, 
        stage=stage, grate=grate, drate = drate, yearst=yearst, sel = select),
                    vec_results, vec_net)

    st_rc  = any(react_comps)  ? "_RC" : ""
    st_ctg = any(contingens)   ? "N1"  : ""
    st_stg = stage > 1 ? "multi$(stage)" : "stc"

    path = joinpath(dirname(path_pruebas), "$(path_salida)$(st_rc)$(st_ctg)$(st_stg).bson")

    println("=== Guardando resultados en: $path ===")
    @save path vector_total

    return vector_total
end