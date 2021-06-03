module FlockingSIR

export FlockingSIRGroup, update, polarization, save_group_to_files, save_simulation,
        load_acng_from_files, initialize_acng_random

# using link-list Cell Molecular Dynamics, refer to http://cacs.usc.edu/education/cs596/01-1LinkedListCell.pdf
# https://journals.aps.org/pre/pdf/10.1103/PhysRevE.84.040301

using HDF5
import CSV
using NPZ
using Distributions

include("General.jl")
include("CollectiveMotionModel.jl")
using .CollectiveMotionModel: ContinuousCollectiveMotionGroup, save_group_to_files, load_group_from_files
using .General: circ_limited, mod_limited, vectorize, polarization, modC_limited!, linked_list_cell, mod2pi!

mutable struct FlockingSIRGroup{T} <: ContinuousCollectiveMotionGroup
    extent::Real
    particles::Int
    radius::Real
    speed::Real
    mu :: Real
    tau::Real
    k :: Real
    noise::Real
    infecting_rate :: Real
    recovering_rate :: Real
    infected :: Array{Bool,1}
    recovered :: Array{Bool,1}
    angle::Array{T,1}
    location::Array{Complex{T},1}
end


function compute_F(locationi::Complex{<:Real},locationj::Complex{<:Real},extent::Real,diameter::Real,k::Real)
    relative_location = circ_limited(locationj-locationi,extent)
    if abs(relative_location)>diameter
        return 0.0+0.0im
    else
        return k*(abs(relative_location)-diameter)*relative_location/abs(relative_location)/diameter
    end
end

function compute_F(relative_location::Complex{<:Real},diameter::Real,k::Real)
    return k*(abs(relative_location)-diameter)*relative_location/abs(relative_location)/diameter
end

function Infected(infecting_rate::Real)
    rand(Bernoulli(infecting_rate))
end

function update_movement_and_turning_i(i::Int,Movement::AbstractArray{Complex{T},1},Angle_difference::AbstractArray{T,1},locationi::Complex{T},locations::AbstractArray{Complex{T},1},
            angli::T,angl::AbstractArray{T,1},infected::AbstractArray{Bool,1},infected_new::AbstractArray{Int,1},recovered::AbstractArray{Bool,1},nneigh::AbstractArray{Int,1},infecting_rate::Real,box::Int,cell::Matrix{Int},head::Matrix{Int},lscl::Vector{Int},diameter::Real,mu::Real,k::Real,extent::Real) where T <: Real
    if infected[i]
        for w in cell[i,1]-1:cell[i,1]+1
            w = mod_limited(w,box)
            for h in cell[i,2]-1:cell[i,2]+1
                h = mod_limited(h,box)
                j = head[w,h]
                while j != -1
                    if i > j
                        rl = circ_limited(locations[j]-locationi,extent)
                        if abs(rl) <= diameter
                            if mu > 0
                                F = mu * compute_F(rl,diameter,k)
                                Movement[i] += F
                                Movement[j] -= F
                            end
                            Angle_difference[i] += circ_limited(angl[j]-angli,2pi)
                            Angle_difference[j] -= circ_limited(angl[j]-angli,2pi)
                            nneigh[i] += 1
                            nneigh[j] += 1
                            if !infected[j] && !recovered[j]
                                infected_new[j] += sign(Infected(infecting_rate))
                            end
                        end
                    end
                    j = lscl[j]
                end        
            end
        end
    elseif !recovered[i]
        for w in cell[i,1]-1:cell[i,1]+1
            w = mod_limited(w,box)
            for h in cell[i,2]-1:cell[i,2]+1
                h = mod_limited(h,box)
                j = head[w,h]
                while j != -1
                    if i > j
                        rl = circ_limited(locations[j]-locationi,extent)
                        if abs(rl) <= diameter
                            if mu > 0
                                F = mu * compute_F(rl,diameter,k)
                                Movement[i] += F
                                Movement[j] -= F
                            end
                            Angle_difference[i] += circ_limited(angl[j]-angli,2pi)
                            Angle_difference[j] -= circ_limited(angl[j]-angli,2pi)
                            nneigh[i] += 1
                            nneigh[j] += 1
                            if infected[j]
                                infected_new[i] += sign(Infected(infecting_rate))
                            end
                        end
                    end
                    j = lscl[j]
                end        
            end
        end
    else
        for w in cell[i,1]-1:cell[i,1]+1
            w = mod_limited(w,box)
            for h in cell[i,2]-1:cell[i,2]+1
                h = mod_limited(h,box)
                j = head[w,h]
                while j != -1
                    if i > j
                        rl = circ_limited(locations[j]-locationi,extent)
                        if abs(rl) <= diameter
                            if mu > 0
                                F = mu * compute_F(rl,diameter,k)
                                Movement[i] += F
                                Movement[j] -= F
                            end
                            Angle_difference[i] += circ_limited(angl[j]-angli,2pi)
                            Angle_difference[j] -= circ_limited(angl[j]-angli,2pi)
                            nneigh[i] += 1
                            nneigh[j] += 1
                        end
                    end
                    j = lscl[j]
                end        
            end
        end
    end
end

function recovering!(infected::AbstractArray{Bool,1},recovered::AbstractArray{Bool,1},recovering_rate::Real)
    @simd for i in 1:length(infected)
        if infected[i]
            recover = sign(Infected(recovering_rate))
            if recover
                @inbounds infected[i] = false
                recovered[i] = true
            end
        end
    end
end


function update_location(locations::AbstractArray{Complex{T},1},Movement::AbstractArray{Complex{T},1},particles::Int,timestep::Real) where T <: Real
    @simd for i in 1:particles
        @inbounds locations[i] += Movement[i]*timestep
    end
end

function update_angle(angl::AbstractArray{T,1},Angle_difference::AbstractArray{T,1},nneigh::AbstractArray{Int,1},noise::Real,particles::Int,tau::Real,timestep::Real) where T <: Real
    @simd for i in 1:particles
        @inbounds angl[i] += (Angle_difference[i]/nneigh[i]/tau +noise*randn())*timestep
    end
end


function update(acng::FlockingSIRGroup{T},timestep::Real) where T <: Real
    noise_n = acng.noise/sqrt(timestep)
    Movement = vectorize.(acng.angle) .* acng.speed
    Angle_difference = zeros(T,acng.particles)
    box,cell,head,lscl = linked_list_cell(acng.location,acng.particles,acng.radius,acng.extent)
    infected_new = convert.(Int,acng.infected)
    nneigh = ones(Int,acng.particles)
    infecting_rate = acng.infecting_rate*timestep
    for i = 1:acng.particles
        update_movement_and_turning_i(i,Movement, Angle_difference,acng.location[i],acng.location, acng.angle[i],acng.angle,acng.infected,infected_new,acng.recovered,nneigh,infecting_rate,box,cell,head,lscl,acng.radius,acng.mu,acng.k,acng.extent)
    end
    update_location(acng.location,Movement,acng.particles,timestep)
    update_angle(acng.angle,Angle_difference,nneigh,noise_n,acng.particles,acng.tau,timestep)
    acng.infected .= sign.(infected_new)
    recovering!(acng.infected,acng.recovered,acng.recovering_rate*timestep)
    mod2pi!(acng.angle)
    modC_limited!(acng.location,acng.extent)
end



function save_simulation(acng::FlockingSIRGroup,frames::Int,skip::Int,timestep::Real,savepath::String;T::DataType=Float32, datatype="HDF5")

    @assert datatype in ["HDF5","NPY","CSV"]
    if !ispath(savepath)
        mkpath(savepath)
    end
    History_location = Array{Complex{T},2}(undef,frames,acng.particles)
    History_angle = Array{T,2}(undef,frames,acng.particles)
    History_infected = Array{Int16,2}(undef,frames,acng.particles)
    History_recovered = Array{Int16,2}(undef,frames,acng.particles)
    History_location[1,:] .= acng.location
    History_angle[1,:] .= acng.angle
    History_infected[1,:] = acng.infected
    History_recovered[1,:] = acng.recovered
    actual_frames = frames
    for t in 2:frames
        for i in 1:skip
            if i == skip
                History_location[t,:] = acng.location
                History_angle[t,:] = acng.angle
                History_infected[t,:] = acng.infected
                History_recovered[t,:] = acng.recovered
            end
            update(acng,timestep)
        end
        if sum(acng.infected) == 0
            actual_frames = t
            break
        end
    end
    pola = Array{T,1}(undef,actual_frames)
    for i in 1:actual_frames
        pola[i] = polarization(History_angle[i,:])
    end
    if datatype == "CSV"
        locationx = real.(History_location[1:actual_frames,:])
        locationy = imag.(History_location[1:actual_frames,:])
        CSV.write(savepath*"/history_locationx.csv",locationx)
        CSV.write(savepath*"/history_locationy.csv",locationy)
        CSV.write(savepath*"/history_angle.csv",History_angle[1:actual_frames,:])
        CSV.write(savepath*"/history_infected.csv",History_infected[1:actual_frames,:])
        CSV.write(savepath*"/history_recovered.csv",History_recovered[1:actual_frames,:])
        CSV.write(savepath*"/history_polarization.csv",pola)
    elseif datatype == "HDF5"
        locationx = real.(History_location[1:actual_frames,:])
        locationy = imag.(History_location[1:actual_frames,:])
        h5open("$savepath/DATA.h5", "w") do file
            write(file,"timestep", timestep)
            write(file, "model type", "flockingSIR")
            write(file, "frames", actual_frames)
            write(file, "skip", skip)
            write(file, "locationx",locationx)
            write(file, "locationy",locationy)
            write(file, "angle",History_angle[1:actual_frames,:])
            write(file, "infected",History_infected[1:actual_frames,:])
            write(file, "recovered",History_recovered[1:actual_frames,:])
            write(file, "polarization",pola)
            write(file, "boundary condition",boundary_condition)
        end
    elseif datatype == "NPY"
        locationx = real.(History_location[1:actual_frames,:])
        locationy = imag.(History_location[1:actual_frames,:])
        npzwrite(savepath*"/history_locationx.npy",locationx)
        npzwrite(savepath*"/history_locationy.npy",locationy)
        npzwrite(savepath*"/history_angle.npy",History_angle[1:actual_frames,:])
        npzwrite(savepath*"/history_infected.npy",History_infected[1:actual_frames,:])
        npzwrite(savepath*"/history_recovered.npy",History_recovered[1:actual_frames,:])
        npzwrite(savepath*"/history_polarization.npy",pola)
    else
        error("datatype can only be \"HDF5\" or \"NPY\" or \"CSV\" ")
    end
end

function load_acng_from_files(loadpath)
    load_group_from_files(FlockingSIRGroup,loadpath)
end

function initialize_acng_random(particles::Int,extent::Real,radius::Real,speed::Real,mu::Real,tau::Real,noise::Real,k::Real)
    group = FlockingSIRGroup(extent,particles,radius,speed,mu,tau,k,noise,rand(particles)*2pi,complex.(rand(particles)*extent,rand(particles)*extent))
    return group
end

end