using  Distributed
addprocs(0)

@everywhere include("sim_reinforce.jl")

@everywhere ACOPF_Extensions.silence_acopf_logs()

syst = "case118_cost_R_D_r"
path_pruebas="test/exp_2025-11-19_170143/PruebasREINFORCE.txt" 
run_rl_reinforce_train("case/$syst", false, false, path_pruebas)

#garverQ
#case24IEEE_P_Glo_r
#case118_cost_R_D_r
# run_rl_reinforce_train("case/case24IEEE_P_Glo_r", false, false)

# syst = "garverQ"
# path_pruebas="test/exp_2025-11-16_190617/PruebasREINFORCE.txt"
# path_salida="Resultados_$(syst)_Din3.bson"
# sistemas = ["case/$syst"]
# react_comps = [false, true]
# contingens = [false]

# res = run_reinforce_evaluation(path_pruebas,path_salida,sistemas,react_comps,contingens)