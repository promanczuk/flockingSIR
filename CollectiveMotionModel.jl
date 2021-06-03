module CollectiveMotionModel

include("General.jl")
using .General: polarization, linked_list_cell
using NPZ


abstract type CollectiveMotionGroup end

abstract type ContinuousCollectiveMotionGroup <: CollectiveMotionGroup end

abstract type DecreteCollectiveMotionGroup <: CollectiveMotionGroup end

function save_group_to_files(group::G,path::String) where G <: CollectiveMotionGroup
    if !ispath(path)
        mkpath(path)
    end
    fn = fieldnames(typeof(group))
    fi = open(path*"/parameters.txt","w")
    for k in 1:length(fn)
        v = getfield(group,k)
        if typeof(v) <: AbstractArray
            npzwrite("$path/$(fn[k]).npy",v)
        else
            if typeof(v) <: String
                write(fi,"$(fn[k])=\"$v\"\n")
            else
                write(fi,"$(fn[k])=$v\n")
            end
        end
    end
    close(fi)
end

function load_group_from_files(Grouptype,path::String)
    fn = fieldnames(Grouptype)
    ft = fieldtypes(Grouptype)
    para = readlines(path*"/parameters.txt")    
    for str in para
        eval(Meta.parse(str))
    end
    for k in 1:length(fn)
        if ft[k] <: AbstractArray
            global tem = npzread(path*"/$(fn[k]).npy")
            eval(Meta.parse("$(fn[k])=tem"))
        end
    end
    return Grouptype(eval.(Meta.parse.(String.(fn)))...)
end

end