using Parameters, Plots #import the libraries we want
include("02Growth_model_stoch.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
@elapsed @time Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim

##############Make plots
#value function
Plots.plot(k_grid, [val_func[:,1] val_func[:,2]], title="Value Function", label=["high z" "low z"], ylabel="Value V(K)", xlabel="Capital")
Plots.savefig("PS1/Graphs/03_Value_Functions.png")

#policy functions
Plots.plot(k_grid, [pol_func[:,1] pol_func[:,2]], title="Policy Functions", label=["high z" "low z"], ylabel="Policy K'(K)", xlabel="Capital")
plot!(k_grid,k_grid,label = "45 degree",color="red",linestyle=:dash)
Plots.savefig("PS1/Graphs/03_Policy_Functions.png")

#changes in policy function
pol_func_δ = copy(pol_func).-k_grid
Plots.plot(k_grid, [pol_func_δ[:,1] pol_func_δ[:,2]], title="Policy Functions Changes", label=["high z" "low z"], ylabel="Savings Policy K'(K)-K", xlabel="Capital")
Plots.savefig("PS1/Graphs/03_Policy_Functions_Changes.png")

println("All done!")
################################
