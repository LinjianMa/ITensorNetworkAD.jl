using ITensors, ChainRulesCore

global AbstractTensor = Union{ITensor}

SetAbstractTensor(t) = (global AbstractTensor = t)

"""
batch_tensor_contraction
"""
function AbstractTensors(tensortype::AbstractTensor, networks::Vector{Vector{ITensor}}; kwargs...)
  dict = Dict{ITensor, tensortype}()
  output_networks = Vector{Vector{tensortype}}()
  for network in networks
    for tensor in network 
      if !haskey(dict, tensor)
        dict[tensor] = tensortype(tensor; kwargs...)
      end
    end
    out_network = Vector{tensortype}([dict[t] for t in network])
    push!(output_networks, out_network)
  end
  return output_networks
end

function abstract_network(tensortype::AbstractTensor, networks::Vector{Vector{ITensor}}, vars; kwargs...)
  tensors = vcat([Vector(vars)], networks)
  abs_tensors = AbstractTensors(tensortype, tensors; kwargs...)
  vars = abs_tensors[1]
  networks = abs_tensors[2:end]
  return networks, Tuple(vars)
end

function ChainRulesCore.rrule(
  ::typeof(abstract_network), tensortype::AbstractTensor, networks::Vector{Vector{ITensor}}, vars; kwargs...
)
  function pullback(v)
    d_networks, dvars = v[1], v[2]
    dvars_itensor = map(x->ITensor(x), dvars)
    return (NoTangent(), NoTangent(), NoTangent(), dvars_itensor)
  end
  return abstract_network(tensortype, networks, vars; kwargs...), pullback
end
