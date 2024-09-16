using Parameters, Plots, LinearAlgebra, Optim, Interpolations #import the libraries we want
include("huggett_model.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
@elapsed @time Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack A_grid = prim

#right now my problem is that the transition matrix is all zeros. There must be something wrong with the row and column indexing. 
##############Make plots
#value function
Plots.plot(A_grid, val_func[:,1], title="Value Function", label="employed", color="cadetblue", linewidth=:2.0, ylabel="Value V(a)", xlabel="assets")
plot!(A_grid, val_func[:,2], label="unemployed", color="palevioletred", linewidth=:2.0)
Plots.savefig("PS2/Graphs/Value_Functions.png")

#policy functions
Plots.plot(A_grid, pol_func[:,1], title="Policy Functions", label="employed", color="cadetblue", linewidth=:2.0, ylabel="Policy a'(a)", xlabel="assets")
plot!(A_grid, pol_func[:,2], label="unemployed", color="palevioletred", linewidth=:2.0)
plot!(A_grid,A_grid,label = "45 degree",color="gray40",linestyle=:dash)
Plots.savefig("PS2/Graphs/Policy_Functions.png")

println("All done!")
################################