using ExPlanRL
using Test
using BSON: @load

@testset "ExPlanRL.jl" begin
    system = joinpath(@__DIR__,"..", "case", "garverQ")
    vector_total = []
    @load joinpath(@__DIR__,"..","test" ,"resultados_500_2025-11-25_171102_3.bson") policy_model timeTrain params nepi perdidas_por_batch VFO semilla recompensas_episodios
    
    push!(vector_total, (policy_model, params, perdidas_por_batch, recompensas_episodios, VFO))

    @testset "evaluar_sistemas: interface tests" begin
        
                
        result = ExPlanRL.evaluar_sistemas(vector_total, [system], [true, false], [false])

        _,_,_,fobj1,time1,plan1,val1 = result[1][1]
        _,_,_,fobj2,time2,plan2,val2 = result[1][2]

        @test isa(val1, Bool)
        @test isa(fobj1, Real)
        @test isa(time1, Real)
        @test isa(plan1, Matrix)

        @test !isnan(fobj1)
        @test !isempty(plan1)
    end
end
