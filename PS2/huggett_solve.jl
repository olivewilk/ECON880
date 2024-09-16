using Parameters, Plots, LinearAlgebra #import the libraries we want
include("huggett_model.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
@elapsed @time Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack A_grid = prim

#right now my problem is that the transition matrix is all zeros. There must be something wrong with the row and column indexing.
##############Make plots
#value function
Plots.plot(A_grid, [val_func[:,1] val_func[:,2]], title="Value Function", label=["employed" "unemployed"], ylabel="Value V(a)", xlabel="assets")
Plots.savefig("PS2/Graphs/Value_Functions.png")

#policy functions
Plots.plot(A_grid, [pol_func[:,1] pol_func[:,2]], title="Policy Functions", label=["employed" "unemployed"], ylabel="Policy a'(a)", xlabel="assets")
plot!(A_grid,A_grid,label = "45 degree",color="red",linestyle=:dash)
Plots.savefig("PS2/Graphs/Policy_Functions.png")

#changes in policy function
#pol_func_δ = copy(pol_func).-k_grid
#Plots.plot(k_grid, [pol_func_δ[:,1] pol_func_δ[:,2]], title="Policy Functions Changes", label=["high z" "low z"], ylabel="Savings Policy K'(K)-K", xlabel="Capital")
#Plots.savefig("PS1/Graphs/03_Policy_Functions_Changes.png")

println("All done!")
################################