
using Distributed
addprocs(4)
@everywhere using Parameters, Plots, SharedArrays #import the libraries we want
include("parallel_functions_stoch.jl") #import the functions that solve our growth model
@everywhere prim, res = Initialize() #initialize primitive and results structs
@time Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim

##############Make plots
#value function
plot(k_grid,  val_func[:,1], title="Value Function", label="high z=1.25", color="cadetblue", linewidth=:2.0, ylabel="Value V(K)", xlabel="Capital")
plot!(k_grid, val_func[:,2], label="low z=.02", color="palevioletred", linewidth=:2.0)
savefig("PS1/Graphs/04_Value_Functions.png")

#policy functions
plot(k_grid, pol_func[:,1], title="Policy Functions", label="K'(K,Zₕ)" , color="cadetblue", linewidth=:2.0, ylabel="Policy K'(K)", xlabel="Capital")
plot!(k_grid, pol_func[:,2], label="K'(K,Zₗ)", color="palevioletred", linewidth=:2.0)
plot!(k_grid,k_grid,label = "45 degree",color="gray40",linestyle=:dash, linewidth=:1.5)
savefig("PS1/Graphs/04_Policy_Functions.png")

#changes in policy function
pol_func_δ = pol_func.-k_grid
plot(k_grid, pol_func_δ[:,1], title="Policy Functions Changes", label="high z=1.25", color="cadetblue", linewidth=:2.0, ylabel="Savings Policy K'(K)-K", xlabel="Capital")
plot!(k_grid, pol_func_δ[:,2], label="low z=.02", color="palevioletred", linewidth=:2.0)
savefig("PS1/Graphs/04_Policy_Functions_Changes.png")

println("All done!")
################################
