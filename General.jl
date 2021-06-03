module General

using Statistics

function complex2tuple(com::Complex)
    return real(com),imag(com)
end

function complexs2array(coms::AbstractArray{T,1}) where T <: Complex
    Array(transpose(hcat(real.(coms),imag.(coms))))
end


function circ(array::Real, extent::Real)
    array = mod(array,extent)
    if array > extent/2
        return array - extent
    else
        return array
    end
end


function circ(array::Complex{<:Real}, extent::Real)
    return complex(circ(real(array),extent),circ(imag(array),extent))
end

function circ(array::Complex{<:Real}, extentx::Real,extenty::Real)
    return complex(circ(real(array),extentx),circ(imag(array),extenty))
end


function circ_limited(array::Complex{<:Real}, extent::Real)    
    vecr = mod_limited(real(array), extent)
    veci = mod_limited(imag(array), extent)
    return complex(circ_limited(vecr,extent),circ_limited(veci,extent))
end

function circ_limited(x::Real, extent::Real)
    x = mod_limited(x,extent)
    if x > extent/2
        return x - extent
    else
        return x
    end
end

function circ_limited(array::Complex{<:Real}, extentx::Real,extenty::Real)    
    vecr = mod_limited(real(array), extentx)
    veci = mod_limited(imag(array), extenty)
    array = complex(circ_limited(vecr,extentx),circ_limited(veci,extenty))
    return array
end


function mod_limited(num::Real,extent::Real)
    # println("num $num extent $extent")
    if num <= 0
        return mod_limited(num+extent,extent)
    elseif num > extent
        return mod_limited(num-extent,extent)
    else
        return num
    end
end

function modC_limited(num::Complex{<:Real},extent::Real)
    # mod for Complex number
    ima = mod_limited(imag(num),extent)
    rea = mod_limited(real(num),extent)
    return complex(rea,ima)
end


function modC_limited!(array::AbstractArray{Complex{T}},extent::Real) where T <: Real
    # mod for complex number, operate on a array directly(modify the array directly)
    for t in 1:length(array)
        @inbounds array[t] = modC_limited(array[t],extent)
    end
end

function modC_limited(num::Complex{<:Real},extentx::Real,extenty::Real)
    # mod for Complex number
    rea = mod_limited(real(num),extentx)
    ima = mod_limited(imag(num),extenty)
    return complex(rea,ima)
end


function modC_limited!(array::AbstractArray{Complex{T}},extentx::Real,extenty::Real) where T <: Real
    # mod for complex number, operate on a array directly(modify the array directly)
    for t in (1:length(array)) :: UnitRange{Int}
        @inbounds array[t] = modC_limited(array[t],extentx,extenty)
    end
end
function mod2pi!(angl::AbstractArray{T}) where T <: Real
    @simd for t in (1:length(angl)) :: UnitRange{Int}
        @inbounds angl[t] = mod2pi(angl[t])
    end
end


function polarization(angles::Array{<:Real,1})
    direc = 0. + .0im
    @simd for i in angles :: AbstractArray{<:Real,1}
        @inbounds direc += exp(i*im)
    end
    return abs(direc/length(angles))
end


function vectorize(num::Real)
    return exp(num*im)
end

function relative_position_ij(locationi::Complex{T},locationj::Complex{T},extent::Real) where T <: Real
    return circ_limited(locationj-locationi,extent)
end



function linked_list_cell(location::Array{Complex{T},1},particles::Int,radius::Real,extent::Real) where T <: Real
    box = Int(floor(extent/radius))                   
                                               
    cell = -ones(Int,particles,2)                   
    head = -ones(Int,box,box)                  
    lscl = -ones(Int,particles)               
    for i in (1:particles) :: UnitRange{Int}                              
        @inbounds cell[i,1] = Int(ceil(real(location[i])/extent*box)) 
        @inbounds cell[i,2] = Int(ceil(imag(location[i])/extent*box))
        if head[cell[i,1],cell[i,2]] == -1                
            @inbounds head[cell[i,1],cell[i,2]] = i
        else
            @inbounds j = head[cell[i,1],cell[i,2]]           
            while lscl[j]!=-1
                @inbounds j = lscl[j]
            end
            @inbounds lscl[j] = i
        end
    end
    return box,cell,head,lscl
end




function copy_struct(struc)
    names = fieldnames(typeof(struc))
    parameters = Array{Any,1}(undef,length(names))
    for i in 1:length(names)
        parameters[i] = getfield(struc,names[i])
    end
    return (typeof(struc))(parameters...)
end





end