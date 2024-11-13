#=
Author: Olivia 
Date: September 2024 
=#

using Parameters, Plots, LinearAlgebra, Optim, Interpolations #import the libraries we want
using Trapz
include("Hopenhayn_model.jl") #import the functions that solve our growth model
prim, resvectors, resfloats = Initialize() #initialize primitive and results structs

pol_func_matrix = zeros(prim.ns, 3)

α=0 #benchmark case
c_f = 10 #lower fixed cost case
@elapsed @time Price_solve(prim, resvectors, resfloats, α, c_f) #solve the model!
@elapsed @time New_entrants(prim, resvectors, resfloats) #solve the model!

@unpack N, W, pol_func, mu_dist = resvectors
@unpack p, M, Ld, Ls, Π= resfloats
@unpack ns, s_grid, v_entrant= prim

#1) Compute model moments 
println("Benchmark")
println("Price Level is $p")
M_inc = sum((1 .- pol_func) .* mu_dist[1:ns])
println("Mass of incumbent firms is $M_inc")
println("Mass of entrants is $M")
M_exit = sum(pol_func .* mu_dist[1:ns])
println("Mass of exits is $M_exit")
println("Aggregate labor is $Ld")
L_incumbent = sum(N.*mu_dist)
println("Labor of incumbents is $L_incumbent")
L_entrant = sum(M.*N.*v_entrant)
println("Labor of entrants is $L_entrant")
println("Fraction of labor in entrants is $(L_entrant/(L_incumbent + L_entrant))")

pol_func_matrix[:,1] = pol_func

##############################################################################
include("Hopenhayn_model.jl") #import the functions that solve our growth model
prim, resvectors, resfloats = Initialize() #initialize primitive and results structs
α=1 #big shock case
c_f = 10 #lower fixed cost case
@elapsed @time Price_solve(prim, resvectors, resfloats, α, c_f) #solve the model!
@elapsed @time New_entrants(prim, resvectors, resfloats) #solve the model!

@unpack N, W, pol_func, mu_dist = resvectors
@unpack p, M, Ld, Ls, Π= resfloats
@unpack ns, s_grid, v_entrant= prim

#1) Compute model moments 
println("alpha is $α")
println("Price Level is $p")
M_inc = sum((1 .- pol_func) .* mu_dist[1:ns])
println("Mass of incumbent firms is $M_inc")
println("Mass of entrants is $M")
M_exit = sum(pol_func .* mu_dist[1:ns])
println("Mass of exits is $M_exit")
println("Aggregate labor is $Ld")
L_incumbent = sum(N.*mu_dist)
println("Labor of incumbents is $L_incumbent")
L_entrant = sum(M.*N.*v_entrant)
println("Labor of entrants is $L_entrant")
println("Fraction of labor in entrants is $(L_entrant/(L_incumbent + L_entrant))")

pol_func_matrix[:,2] = pol_func

##############################################################################
include("Hopenhayn_model.jl") #import the functions that solve our growth model
prim, resvectors, resfloats = Initialize() #initialize primitive and results structs
α=2 #smaller shock case
c_f = 10 #lower fixed cost case
@elapsed @time Price_solve(prim, resvectors, resfloats, α, c_f) #solve the model!
@elapsed @time New_entrants(prim, resvectors, resfloats) #solve the model!
@unpack N,mu_dist, W, pol_func = resvectors
@unpack p, M, Ld, Ls, Π= resfloats
@unpack ns, s_grid, v_entrant= prim

#1) Compute model moments 
println("alpha is $α")
println("Price Level is $p")
M_inc = sum((1 .- pol_func) .* mu_dist[1:ns])
println("Mass of incumbent firms is $M_inc")
println("Mass of entrants is $M")
M_exit = sum(pol_func .* mu_dist[1:ns])
println("Mass of exits is $M_exit")
println("Aggregate labor is $Ld")
L_incumbent = sum(N.*mu_dist)
println("Labor of incumbents is $L_incumbent")
L_entrant = sum(M.*N.*v_entrant)
println("Labor of entrants is $L_entrant")
println("Fraction of labor in entrants is $(L_entrant/(L_incumbent + L_entrant))")

pol_func_matrix[:,3] = pol_func


#2) Plot exit decision rules 
state = [1,2,3,4,5]
Plots.plot(state, pol_func_matrix[:,1], title="Exit Decision", color="cadetblue", label="benchmark", linewidth=:2.0, xlabel="Productivity State")
plot!(state, pol_func_matrix[:,2], color="palevioletred", label="α=1", linewidth=:2.0)
plot!(state, pol_func_matrix[:,3], color="mistyrose3", label="α=2", linewidth=:2.0)
Plots.savefig("PS3/Graphs/Policy_Function_cf10.png")

#3) How does exit decision rule change if you increase fixed costs from 10 to 15?
include("Hopenhayn_model.jl") #import the functions that solve our growth model
prim, resvectors, resfloats = Initialize() #initialize primitive and results structs
α=0 #benchmark case
c_f = 15 #lower fixed cost case
@elapsed @time Price_solve(prim, resvectors, resfloats, α, c_f) #solve the model!
@elapsed @time New_entrants(prim, resvectors, resfloats) #solve the model!
@unpack N,mu_dist, W, pol_func = resvectors
@unpack p, M, Ld, Ls, Π= resfloats
@unpack ns, s_grid, v_entrant= prim
pol_func_matrix[:,1] = pol_func

prim, resvectors, resfloats = Initialize() #initialize primitive and results structs
α=1 #larger shock case
c_f = 15 #lower fixed cost case
@elapsed @time Price_solve(prim, resvectors, resfloats, α, c_f) #solve the model!
@elapsed @time New_entrants(prim, resvectors, resfloats) #solve the model!
@unpack N,mu_dist, W, pol_func = resvectors
@unpack p, M, Ld, Ls, Π= resfloats
@unpack ns, s_grid, v_entrant= prim
pol_func_matrix[:,2] = pol_func

prim, resvectors, resfloats = Initialize() #initialize primitive and results structs
α=2 #smaller shock case
c_f = 15 #lower fixed cost case
@elapsed @time Price_solve(prim, resvectors, resfloats, α, c_f) #solve the model!
@elapsed @time New_entrants(prim, resvectors, resfloats) #solve the model!
@unpack N,mu_dist, W, pol_func = resvectors
@unpack p, M, Ld, Ls, Π= resfloats
@unpack ns, s_grid, v_entrant= prim
pol_func_matrix[:,3] = pol_func


#Plot exit decision rules 
state = [1,2,3,4,5]
Plots.plot(state, pol_func_matrix[:,1], title="Exit Decision", color="cadetblue", label="benchmark", linewidth=:2.0, xlabel="Productivity State")
plot!(state, pol_func_matrix[:,2], color="palevioletred", label="α=1", linewidth=:2.0)
plot!(state, pol_func_matrix[:,3], color="mistyrose3", label="α=2", linewidth=:2.0)
Plots.savefig("PS3/Graphs/Policy_Function_cf15.png")