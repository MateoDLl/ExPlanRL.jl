
function prepare_case(system::String, shuntcomp::Bool, n1ctg::Bool;
                      stage::Int = 1, grate::Float64 = 20.0,
                       drate::Float64 = 10.0, yearst::Int = 1)

    caseStudyData = ACOPF_Extensions.setup_case(system, shuntcomp, n1ctg,
                Stage=stage, growth_rate=grate, d_rate = drate, years_stage =yearst)
    caseStudyData["Mat_cost"] = calc_mat_cost(caseStudyData)
    caseStudyData["Idx_Cost_Cap"] = calc_mat_capacity(caseStudyData)
    Mat_cost = caseStudyData["Mat_cost"]

    # Extract data for scaling
    discount = caseStudyData["Discount"]'
    growth   = caseStudyData["Growth"]'
    stages   = caseStudyData["Stage"]

    # Compute normalized cost and capacity matrices
    caseStudyData["Mat_Cap_U"]  = maximum(caseStudyData["Idx_Cost_Cap"]) ./ 
                                  (caseStudyData["Idx_Cost_Cap"] .* ones(1, stages))
    caseStudyData["Mat_Cap_U"]  ./= maximum(caseStudyData["Mat_Cap_U"])
    cost_vals = vec(Mat_cost)

    unique_vals = unique(cost_vals)
    sort!(unique_vals, rev = true)
    n = length(unique_vals)

    rank_map = Dict(v => 0.5+k/(2*n+1) for (k,v) in enumerate(unique_vals))
    rank_norm = [rank_map[v] for v in cost_vals]

    Mat_cost_rank = reshape(rank_norm, size(Mat_cost))

    caseStudyData["Mat_cost_U"] = (Mat_cost_rank .* discount) ./ growth
    
    return caseStudyData
end

function get_experiment_folder(timestamp;base="experiments")        
    folder = "$base/exp_$timestamp"
    mkpath(folder)
    return folder
end

"""
    calc_mat_cost(data::Dict)

Builds a cost vector for all candidate elements (AC lines, DC lines, storage units).

# Arguments
- `data`: Dictionary containing network data and candidate element information.

# Returns
- `Vector{Float64}`: Cost vector (rows = total number of candidates).
"""
function calc_mat_cost(data::Dict)
    n_ac = get(data, "ne_branch", nothing) === nothing ? 0 : length(data["ne_branch"])
    n_dc = get(data, "branchdc_ne", nothing) === nothing ? 0 : length(data["branchdc_ne"])
    n_st = get(data, "storagecost", nothing) === nothing ? 0 : length(data["storagecost"])

    total = n_ac + n_dc + n_st
    mat_cost = zeros(Float64, total)

    # --- AC branches ---
    for k in 1:n_ac
        mat_cost[k] = data["ne_branch"][string(k)]["construction_cost"]
    end

    # --- DC branches ---
    for k in 1:n_dc
        mat_cost[n_ac + k] = data["branchdc_ne"][string(k)]["cost"]
    end

    # --- Storage ---
    for k in 1:n_st
        mat_cost[n_ac + n_dc + k] = data["storagecost"][string(k)]["cost"]
    end

    return reshape(mat_cost, :, 1)
end


"""
    calc_mat_capacity(data::Dict)

Builds the capacity matrix for all candidate components, adjusted by discount rate
and normalized by investment costs.

# Arguments
- `data`: Dictionary containing network data and parameters.

# Returns
- `Matrix{Float64}`: Capacity-to-cost matrix (rows = candidates, cols = stages).
"""
function calc_mat_capacity(data::Dict)
    n_ac = get(data, "ne_branch", nothing) === nothing ? 0 : length(data["ne_branch"])
    n_dc = get(data, "branchdc_ne", nothing) === nothing ? 0 : length(data["branchdc_ne"])
    n_st = (get(data, "storagecost", nothing) === nothing || !get(data, "STORAGE", false)) ? 0 : length(data["storagecost"])

    total = n_ac + n_dc + n_st
    mat_cap = zeros(Float64, total)

    # --- AC branches ---
    for k in 1:n_ac
        mat_cap[k] = data["ne_branch"][string(k)]["rate_a"]
    end

    # --- DC branches ---
    for k in 1:n_dc
        mat_cap[n_ac + k] = data["branchdc_ne"][string(k)]["rateA"]
    end

    # --- Storage ---
    for k in 1:n_st
        mat_cap[n_ac + n_dc + k] = data["storagecost"][string(k)]["cost"]
    end

    # --- Capacity matrix per stage ---
    stages = data["Stage"]
    mat_cost = data["Mat_cost"]
    discounts = data["Discount"]

    cap_cost = zeros(Float64, total, stages)
    for i in 1:stages
        cap_cost[:, i] .= discounts[i] .* mat_cap ./ mat_cost
    end

    return cap_cost
end
