
######## Question 2 answers ########
include("Aiyagari.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
@elapsed @time SolveModel() #solve the model!

@unpack k_policy = res
@unpack k_grid = prim

using Plots
#policy function
Plots.plot(k_grid, k_policy[:,1], title="Policy Function", color="cadetblue", label="employed", linewidth=:2.0)
plot!(k_grid, k_policy[:,2], color="palevioletred", label="unemployed", linewidth=:2.0)
Plots.savefig("Midterm/Graphs/02_Policy_Functions_new.png")

println("Steady state capital stock: ", res.K)
println("All done!")
################################ 

######## Question 3 answers ########
include("Aiyagari_KS.jl") #import the functions that solve our growth model

prim, res, sim, sho = Initialize() #initialize primitive and results structs

#VFI(prim, res, sim, sho; tol=1e-10, max_iter=15000)
#prim.K_grid

@elapsed @time EstimateRegression(prim, res, sim; tol=1e-6)

res.a₀
res.a₁
res.b₀
res.b₁
res.R²_a 
res.R²_b 
#using Plots
Plots.plot(sim.burn+1:sim.T, res.K_estimate, title="Simulated Capital Stock over time", color="cadetblue", linewidth=:2.0)
using StatsBase
mean(res.K_estimate)

Plots.plot(prim.k_grid, res.k_policy[:,1,1,1], title="Policy Function", color="cadetblue", label="employed, good aggregate", linewidth=:2.0)
plot!(prim.k_grid, res.k_policy[:,2,1,1], color="palevioletred", label="unemployed, good aggregate", linewidth=:2.0)
plot!(prim.k_grid, res.k_policy[:,1,1,2], color="blue", label="employed, bad aggregate", linewidth=:2.0)
plot!(prim.k_grid, res.k_policy[:,1,1,2], color="red", label="unemployed, bad aggregate", linewidth=:2.0)
