using ITensors, AutoHOOT

const ad = AutoHOOT.autodiff

function get_symbol(i::Int)
  if i < 26
    return 'a' + i
  elseif i < 52
    return 'A' + i - 26
  end
  return Char(i + 140)
end

"""
Retrieve the key from the dictionary that maps AutoHOOT nodes
to ITensor tensors. Returns Nothing if key not exists.
"""
retrieve_key(dict, value) = findfirst(v -> (v === value), dict)

"""Compute the computational graph defined in AutoHOOT.
Parameters
----------
outnodes: A list of AutoHOOT einsum nodes
node_dict: A dictionary mapping AutoHOOT node to ITensor tensor
Returns
-------
A list of ITensor tensors.
"""
function compute_graph!(out_nodes, node_dict)
  topo_order = ad.find_topo_sort(out_nodes)
  for node in topo_order
    if haskey(node_dict, node) == false && node.name != "1.0"
      input_list = []
      for n in node.inputs
        if haskey(node_dict, n)
          push!(input_list, node_dict[n])
        else
          # the scalar that can neglect
          @assert(n.name == "1.0")
        end
      end
      node_dict[node] = contract(input_list)
    end
  end
  return [node_dict[node] for node in out_nodes]
end

compute_graph(out_nodes, node_dict) = compute_graph!(out_nodes, copy(node_dict))

"""Extract an ITensor network from an input network based on AutoHOOT einsum tree.
The ITensor input network is defined by the tensors in node_dict.
Note: this function ONLY extracts network based on the input nodes rather than einstr.
The output network can be hierarchical.
Parameters
----------
outnode: AutoHOOT einsum node
node_dict: A dictionary mapping AutoHOOT node to ITensor tensor
Returns
-------
A list representing the ITensor output network.
Example
-------
>>> extract_network(einsum("ab,bc->ac", A, einsum("bd,dc->bc", B, C)),
                    Dict(A => A_tensor, B => B_tensor, C => C_tensor))
>>> [A_tensor, [B_tensor, C_tensor]]
"""
function extract_network(out_node, node_dict)
  topo_order = ad.find_topo_sort([out_node])
  node_dict = copy(node_dict)
  for node in topo_order
    if haskey(node_dict, node) == false && node.name != "1.0"
      input_list = []
      for n in node.inputs
        if haskey(node_dict, n)
          push!(input_list, node_dict[n])
        else
          # the scalar that can neglect
          @assert(n.name == "1.0")
        end
      end
      node_dict[node] = input_list
    end
  end
  return node_dict[out_node]
end

# perform inner product between two nodes
function inner(n1, n2)
  @assert(n1.shape == n2.shape)
  str = join([get_symbol(i) for i in 1:length(n1.shape)], "")
  return ad.einsum(str * "," * str * "->", n1, n2)
end

"""Generate AutoHOOT einsum expression based on a list of ITensor input networks
Parameters
----------
network_list: An array of networks. Each network is represented by an array of ITensor tensors
Returns
-------
A list of AutoHOOT einsum node;
A dictionary mapping AutoHOOT input node to ITensor tensor
"""
function generate_einsum_expr(network_list::Array)
  node_dict = Dict()
  outnodes = []
  for network in network_list
    input_nodes = input_nodes_generation!(network, node_dict)
    einstr = einstr_generation(network)
    push!(outnodes, ad.einsum(einstr, input_nodes...))
  end
  return outnodes, node_dict
end

function update_dict!(node_dict::Dict, tensor)
  i = length(node_dict) + 1
  if length(inds(tensor)) != 0
    nodename = "tensor" * string(i)
    shape = [space(index) for index in inds(tensor)]
    node = ad.Variable(nodename; shape=shape)
  else
    node = ad.scalar(scalar(tensor))
  end
  node_dict[node] = tensor
  return node
end

"""Generate AutoHOOT nodes based on ITensor tensors
Parameters
----------
network: An array of ITensor tensors
node_dict: A dictionary mapping AutoHOOT node to ITensor tensor.
node_dict will be inplace updated.
Returns
-------
node_list: An array of AutoHOOT nodes
"""
function input_nodes_generation!(network::Array, node_dict::Dict)
  node_list = []
  for (i, tensor) in enumerate(network)
    node = retrieve_key(node_dict, tensor)
    if node == nothing
      node = update_dict!(node_dict, tensor)
    end
    push!(node_list, node)
  end
  return node_list
end

function einstr_generation(network::Array)
  index_dict = Dict{Index{Int64},Char}()
  label_num = 0
  string_list = []
  # build input string
  for tensor in network
    str = ""
    for index in inds(tensor)
      if haskey(index_dict, index) == false
        index_dict[index] = get_symbol(label_num)
        label_num += 1
      end
      str = str * index_dict[index]
    end
    push!(string_list, str)
  end
  instr = join(string_list, ",")
  # build output string
  output_inds = noncommoninds(network...)
  outstr = join([index_dict[i] for i in output_inds])
  return instr * "->" * outstr
end