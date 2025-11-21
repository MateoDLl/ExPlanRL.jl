function evaluar_parametros(params, semilla, caseData, timeGlobal; policy_model = nothing)
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
    β = params[7]
    # Ejecuta el algoritmo
    hidden1 = 64
    hidden2 = 32
    if isnothing(policy_model)
        policy_model_0 = Flux.Chain(
            Flux.Dense(12, hidden1, relu),
            Flux.Dense(hidden1, hidden2, relu),
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
    timeTrain = @elapsed policy_model = entrenar_reinforce_batch_baseline!(nepi, entorno, policy_model_0, opt, frecuencia, γ, β, perdidas_por_batch, recompensas_episodios,caseData)
    entorno = RedElectricaEntorno(nlines, Stage, vk, vs, caseData)
    
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    wid = Distributed.myid()                        # ID del worker → 2,3,4,...
    #uid = string(UUIDs.uuid4())[1:8]    # 8 caracteres aleatorios

    folder = get_experiment_folder(timeGlobal, base="test")

    filename = joinpath(folder,
                        "resultados_$(nepi)_$(timestamp)_$(wid).bson")

    open(joinpath(folder, "PruebasREINFORCE.txt"), "a") do io
        println(io, basename(filename))
    end


    #@save filename timeTrain params nepi perdidas_por_batch
    VFO = evaluar_red_reinforce(policy_model, entorno, caseData)  
    @save filename policy_model timeTrain params nepi perdidas_por_batch VFO semilla recompensas_episodios  
    # Devuelve el valor a minimizar 
    #return VFO
end

#Random.seed!(1234)
function run_rl_reinforce_train(system::String, rc::Bool, n1::Bool)
    caseStudyData = prepare_case(system, rc, n1)
    p1 = [4]
    p2 = [4]
    p3 = [3 6 9]
    p4 = [0.005 0.01]
    p5 = [0.99 0.999]
    p6 = [500]
    p7 = [0.1]
    correr_experimentos_pmap(p1,p2,p3,p4,p5,p6,p7, caseStudyData)
end

function run_rl_reinforce_train(system::String, rc::Bool, n1::Bool, path::String)
    caseStudyData = prepare_case(system, rc, n1)
    correr_experimentos_trained_pmap(path, caseStudyData)
end

function correr_experimentos(p1, p2, p3, p4, p5, p6, p7, caseStudyData)
    seed = 1000
    combo_id = 0
    for (v1, v2, v3, v4, v5, v6, v7) in Iterators.product(p1, p2, p3, p4, p5, p6, p7)
        for rep in 1:1
            parametros_test = [v1, v2, v3, v4, v5, v6, v7]
            semilla = seed + combo_id * 10 + rep
            Random.seed!(semilla)
            evaluar_parametros(parametros_test, semilla, caseStudyData, timeGlobal)
        end
        combo_id += 1
    end
end

function correr_experimentos_seleccionado(experimentos, p1, p2, p3, p4, p5, p6, p7, caseStudyData)
    seed = 1000
    combo_id = 0
    experimento_global = 1  # contador global

    for (v1, v2, v3, v4, v5, v6, v7) in Iterators.product(p1, p2, p3, p4, p5, p6, p7)
        for rep in 1:1
            if experimento_global in experimentos
                parametros_test = [v1, v2, v3, v4, v5, v6, v7]
                semilla = seed + combo_id * 10 + rep
                Random.seed!(semilla)
                evaluar_parametros(parametros_test, semilla, caseStudyData, timeGlobal)
            end
            experimento_global += 1
        end
        combo_id += 1
    end
end

function wrapper(parametros_test, semilla, caseStudyData, timeGlobal; policy=nothing)
    Random.seed!(semilla)
    evaluar_parametros(parametros_test, semilla, caseStudyData, timeGlobal, policy_model = policy)
end


function correr_experimentos_pmap(p1,p2,p3,p4,p5,p6,p7, caseStudyData)
    seed = 1000
    trabajos = []
    timeGlobal = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    combo_id = 0
    for (v1, v2, v3, v4, v5, v6, v7) in Iterators.product(p1, p2, p3, p4, p5, p6, p7)
        for rep in 1:1
            parametros_test = (v1, v2, v3, v4, v5, v6, v7)
            semilla = seed + combo_id * 10 + rep
            push!(trabajos, (parametros_test, semilla))
        end
        combo_id += 1
    end
    Distributed.pmap(trabajos) do (parametros_test, semilla) 
        wrapper(parametros_test, semilla, caseStudyData, timeGlobal)
    end
    
end 

function wrapper_pmap(args)
    parametros_test, semilla, policy_model = args
    return wrapper(parametros_test, semilla, caseStudyData, timeGlobal, policy = policy_model)
end

function correr_experimentos_trained_pmap(path_archivo, caseStudyData)
    trabajos = []
    timeGlobal = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

    folder = dirname(path_archivo)

    open(path_archivo, "r") do archivo
        for (id, filename) in enumerate(eachline(archivo))
            @load joinpath(folder, filename) policy_model timeTrain params nepi perdidas_por_batch VFO semilla recompensas_episodios
            push!(trabajos, (params, semilla, policy_model) )
        end
    end
    Distributed.pmap(trabajos) do (parametros_test, semilla, policy_model)
        wrapper(parametros_test, semilla, caseStudyData, timeGlobal, policy = policy_model)
    end
end  

function evaluar_parametros(params, policy_model, nlines, Stage, caseStudyData)
    vk = round(Int,params[1])
    vs = round(Int,params[2])
    
    entorno = RedElectricaEntorno(nlines, Stage, vk, vs, caseStudyData)
    VFO, top = evaluar_red_reinforce(policy_model, entorno, caseStudyData)  
    return VFO, top
end


function cargar_modelos(path_archivo, vec_id)
    vec_names = String[]
    vec_results = []
    vec_VFO = []
    folder = dirname(path_archivo)

    open(path_archivo, "r") do archivo
        for (id, filename) in enumerate(eachline(archivo))
            if id in vec_id
                push!(vec_names, filename)
                @load joinpath(folder, filename) policy_model timeTrain params nepi perdidas_por_batch VFO semilla recompensas_episodios
                push!(vec_results, (policy_model, params, perdidas_por_batch, recompensas_episodios, VFO))
                push!(vec_VFO, VFO)
                plot_perd_reward(perdidas_por_batch, recompensas_episodios, VFO, id, folder)
            end
        end
    end

    return vec_names, vec_results, vec_VFO
end

function evaluar_sistemas(vec_results, vec_id, sistemas, react_comps, contingens;
                            stage::Int = 1, grate::Float64 = 20.0,
                                  drate::Float64 = 10.0, yearst::Int = 1)
    vector_total = []

    for (i, sis_train) in enumerate(vec_results)
        vectorRes = []

        for (v1, v2, v3) in Iterators.product(sistemas, react_comps, contingens)
            println("Sistema:$v1, RC:$v2, Ctg:$v3")
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
            )

            valido = true
            if isnothing(valor)
                valor = sum(caseStudyData["Mat_cost"] .* top)
                valido = false
            end

            push!(vectorRes, (v1, v2, v3, round(valor, digits=2),
                              round(time_test, digits=2), top, valido, vec_id[i]))
        end

        push!(vector_total, vectorRes)
    end

    return vector_total
end

# Para que las funciones y variables necesarias estén disponibles en todos los workers
function evaluar_sistemas_worker(sis_train, id, sistemas, react_comps, contingens;
                                             stage::Int=1, grate::Float64=20.0,
                                             drate::Float64=10.0, yearst::Int=1)
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
        )

        valido = true
        if isnothing(valor)
            valor = sum(caseStudyData["Mat_cost"] .* top)
            valido = false
        end

        push!(vectorRes, (v1, v2, v3, round(valor, digits=2),
                          round(time_test, digits=2), top, valido, id))
    end

    return vectorRes
end



"""
    run_reinforce_evaluation(path_pruebas, path_salida,
                             sistemas = ["garverQ", "case24IEEE_P_Glo_r", "case118_cost_R_D_r"],
                             react_comps = [false, true],
                             contingens = [false];
                             vec_id = nothing)

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
                                  drate::Float64 = 10.0, yearst::Int = 1)

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
    vec_names, vec_results, vec_VFO = cargar_modelos(path_pruebas, vec_id)

    println("=== Ejecutando evaluación sobre sistemas ===")

    # Usando pmap
    vector_total = pmap((sis_train, id) -> evaluar_sistemas_worker(sis_train, id, sistemas, react_comps, contingens, 
        stage=stage, grate=grate, drate = drate, yearst=yearst),
                    vec_results, vec_id)
    # vector_total = evaluar_sistemas(
    #     vec_results,
    #     vec_id,
    #     sistemas,
    #     react_comps,
    #     contingens,
    # )

    println("=== Guardando resultados en: $path_salida ===")
    @save path_salida vector_total

    return vector_total
end