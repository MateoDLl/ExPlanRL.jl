using ExPlanRL
using Test

@testset "ExPlanRL.jl" begin
    # Write your tests here.
    system = joinpath(@__DIR__,"..", "case", "garverQ")
    BSON.@load "resultados_500_2025-11-25_171102_3.bson" vector_total

    @testset "evaluar_sistemas: interface tests" begin
        
                
        result = evaluar_sistemas(vector_total, [system], [true, false], [false])

        _,_,_,fobj1,time1,plan1,val1 = result[1]
        _,_,_,fobj2,time2,plan2,val2 = result[2]

        @test isa(val1, Bool)
        @test isa(fobj1, Real)
        @test isa(time1, Real)
        @test isa(plan1, Matrix)

        @test !isnan(fobj1)
        @test !isempty(plan1)
    end
end
