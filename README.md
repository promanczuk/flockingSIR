# flockingSIR
Simulation code for model of contagion dynamics in systems of self-propelled particles with alignment.

flockingSIR.jl contains the package of the model.

General.jl and CollectiveMotionModel.jl are supporting modules.

To using the flockingSIR package, just

```julia
including("savepath/flockingSIR.jl")
using Main.FlockingSIR
```
Use 
```julia
x = FlockingSIRGroup(...)
```
to initialize a group x, and 
```julia
update(x,timestep)
```
to update the group with a timestep.

The state of group x can be saved to path by
```julia
save_group_to_files(x,path)
```
A group can be loaded with saved files under path by
```julia
load_fsg_from_files(path)
```
Simulation data is saved with
```julia
save_simulation(...)
```
