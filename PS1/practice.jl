z_g=2
z_b=1
# if rand()>=0.5
#     z= z_g
# else 
#     z=z_b
# end 
z= z_b
pi=rand()
if pi<=0.977 && z==z_g
    z = z_g
elseif pi>.977 && z==z_g 
    z = z_b
elseif pi<=0.074 && z==z_b
    z = z_g
elseif pi>0.074 && z==z_b
    z = z_b
end 

println(z)