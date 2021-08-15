using Random
using .Models

"""
A finite size PEPS type.
"""
struct PEPS
  data::Matrix{ITensor}
end

PEPS(Nx::Int, Ny::Int) = PEPS(Matrix{ITensor}(undef, Nx, Ny))

"""
    PEPS([::Type{ElT} = Float64, sites; linkdims=1)
Construct an PEPS filled with Empty ITensors of type `ElT` from a collection of indices.
Optionally specify the link dimension with the keyword argument `linkdims`, which by default is 1.
"""
function PEPS(::Type{T}, sites::Matrix{<:Index}; linkdims::Integer=1) where {T<:Number}
  Ny, Nx = size(sites)
  tensor_grid = Matrix{ITensor}(undef, Ny, Nx)
  # we assume the PEPS at least has size (2,2). Can generalize if necessary
  @assert(Nx >= 2 && Ny >= 2)

  lh = Matrix{Index}(undef, Ny, Nx - 1)
  for ii in 1:(Nx - 1)
    for jj in 1:(Ny)
      lh[jj, ii] = Index(linkdims, "Lh,$jj,$ii")
    end
  end
  lv = Matrix{Index}(undef, Ny - 1, Nx)
  for ii in 1:(Nx)
    for jj in 1:(Ny - 1)
      lv[jj, ii] = Index(linkdims, "Lv,$jj,$ii")
    end
  end

  # boundary cases
  tensor_grid[1, 1] = ITensor(T, lh[1, 1], lv[1, 1], sites[1, 1])
  tensor_grid[1, Nx] = ITensor(T, lh[1, Nx - 1], lv[1, Nx], sites[1, Nx])
  tensor_grid[Ny, 1] = ITensor(T, lh[Ny, 1], lv[Ny - 1, 1], sites[Ny, 1])
  tensor_grid[Ny, Nx] = ITensor(T, lh[Ny, Nx - 1], lv[Ny - 1, Nx], sites[Ny, Nx])
  for ii in 2:(Nx - 1)
    tensor_grid[1, ii] = ITensor(T, lh[1, ii], lh[1, ii - 1], lv[1, ii], sites[1, ii])
    tensor_grid[Ny, ii] = ITensor(
      T, lh[Ny, ii], lh[Ny, ii - 1], lv[Ny - 1, ii], sites[Ny, ii]
    )
  end

  # inner sites
  for jj in 2:(Ny - 1)
    tensor_grid[jj, 1] = ITensor(T, lh[jj, 1], lv[jj, 1], lv[jj - 1, 1], sites[jj, 1])
    tensor_grid[jj, Nx] = ITensor(
      T, lh[jj, Nx - 1], lv[jj, Nx], lv[jj - 1, Nx], sites[jj, Nx]
    )
    for ii in 2:(Nx - 1)
      tensor_grid[jj, ii] = ITensor(
        T, lh[jj, ii], lh[jj, ii - 1], lv[jj, ii], lv[jj - 1, ii], sites[jj, ii]
      )
    end
  end

  return PEPS(tensor_grid)
end

PEPS(sites::Matrix{<:Index}, args...; kwargs...) = PEPS(Float64, sites, args...; kwargs...)

function Random.randn!(P::PEPS)
  randn!.(P.data)
  normalize!.(P.data)
  return P
end

Base.:+(A::PEPS, B::PEPS) = broadcast_add(A, B)

broadcast_add(A::PEPS, B::PEPS) = PEPS(A.data .+ B.data)

broadcast_minus(A::PEPS, B::PEPS) = PEPS(A.data .- B.data)

broadcast_mul(c::Number, A::PEPS) = PEPS(c .* A.data)

broadcast_inner(A::PEPS, B::PEPS) = mapreduce(v -> v[], +, A.data .* B.data)

ITensors.prime(P::PEPS, n::Integer=1) = PEPS(map(x -> prime(x, n), P.data))

# prime a PEPS with specified indices
function ITensors.prime(indices::Array{<:Index,1}, P::PEPS, n::Integer=1)
  function primeinds(tensor)
    prime_inds = [ind for ind in inds(tensor) if ind in indices]
    return replaceinds(tensor, prime_inds => prime(prime_inds, n))
  end
  return PEPS(map(x -> primeinds(x), P.data))
end

# prime linkinds of a PEPS
function ITensors.prime(::typeof(linkinds), P::PEPS, n::Integer=1)
  return PEPS(mapinds(x -> prime(x, n), linkinds, P.data))
end

function ITensors.addtags(::typeof(linkinds), P::PEPS, args...)
  return PEPS(addtags(linkinds, P.data, args...))
end

function ITensors.removetags(::typeof(linkinds), P::PEPS, args...)
  return PEPS(removetags(linkinds, P.data, args...))
end

ITensors.data(P::PEPS) = P.data

split_network(P::PEPS, rotation=false) = PEPS(split_network(data(P); rotation=rotation))

function ITensors.commoninds(p1::PEPS, p2::PEPS)
  return mapreduce(a -> commoninds(a...), vcat, zip(p1.data, p2.data))
end

"""Returns a tree structure for a line of tensors with projectors
Parameters
----------
line_size: size of the line structure
center_index: the center index of the line
site_tensors: a function, site_tensors(i) returns a list of tensors at position i
projectors: the projectors of the line structure
Returns
-------
Two trees, one in front of the center_index and another one after center_index
Example
-------
   |  |   |
p1-p2 |  p3-p4
 | |  |   | |
 | |  |   | |
s1-s2-s3-s4-s5
here line_size=5, center_index=3, si represents the list of tensors returned by site_tensors(i),
projectors are [p1, p2, p3, p4].
Returns two trees: [[s1, p1], s2, p2] and [[s5, p4], s4, p3]
"""
function tree(line_size, center_index, site_tensors, projectors::Vector{ITensor})
  front_tree, back_tree = nothing, nothing
  for i in 1:(center_index - 1)
    connect_projectors = neighboring_tensors(SubNetwork(site_tensors(i)), projectors)
    if front_tree == nothing
      inputs = vcat(site_tensors(i), connect_projectors)
    else
      inputs = vcat(site_tensors(i), connect_projectors, [front_tree])
    end
    front_tree = SubNetwork(inputs)
  end
  for i in line_size:-1:(center_index + 1)
    connect_projectors = neighboring_tensors(SubNetwork(site_tensors(i)), projectors)
    if back_tree == nothing
      inputs = vcat(site_tensors(i), connect_projectors)
    else
      inputs = vcat(site_tensors(i), connect_projectors, [back_tree])
    end
    back_tree = SubNetwork(inputs)
  end
  return front_tree, back_tree
end

function tree(sub_peps_bra::Vector, sub_peps_ket::Vector, projectors::Vector{ITensor})
  out_inds = inds(SubNetwork(vcat(sub_peps_bra, sub_peps_ket, projectors)))
  is_neighbor(t) = length(intersect(out_inds, inds(t))) > 0
  center_index = [i for (i, t) in enumerate(sub_peps_bra) if is_neighbor(t)]
  @assert length(center_index) == 1
  center_index = center_index[1]
  @assert is_neighbor(sub_peps_ket[center_index])
  site_tensors(i) = [sub_peps_bra[i], sub_peps_ket[i]]
  front_tree, back_tree = tree(
    length(sub_peps_bra), center_index, site_tensors, projectors::Vector{ITensor}
  )
  return SubNetwork([
    sub_peps_bra[center_index], sub_peps_ket[center_index], front_tree, back_tree
  ])
end

function tree(
  sub_peps_bra::Vector, sub_peps_ket::Vector, mpo::Vector, projectors::Vector{ITensor}
)
  out_inds = inds(SubNetwork(vcat(sub_peps_bra, sub_peps_ket, mpo, projectors)))
  is_neighbor(t) = length(intersect(out_inds, inds(t))) > 0
  center_index = [i for (i, t) in enumerate(sub_peps_bra) if is_neighbor(t)]
  @assert length(center_index) == 1
  center_index = center_index[1]
  @assert is_neighbor(sub_peps_ket[center_index])
  site_tensors(i) = [sub_peps_bra[i], sub_peps_ket[i], mpo[i]]
  front_tree, back_tree = tree(
    length(sub_peps_bra), center_index, site_tensors, projectors::Vector{ITensor}
  )
  return SubNetwork([
    sub_peps_bra[center_index],
    sub_peps_ket[center_index],
    mpo[center_index],
    front_tree,
    back_tree,
  ])
end

# Get the tensor network of <peps|peps'>
function inner_network(peps::PEPS, peps_prime::PEPS)
  return vcat(vcat(peps.data...), vcat(peps_prime.data...))
end

function inner_network(peps::PEPS, peps_prime::PEPS, projectors::Vector{<:ITensor})
  network = inner_network(peps::PEPS, peps_prime::PEPS)
  return vcat(network, projectors)
end

function inner_network(
  peps::PEPS, peps_prime::PEPS, projectors::Vector{<:ITensor}, ::typeof(tree)
)
  Ny, Nx = size(peps.data)
  function get_tree(i)
    line_tensors = vcat(peps.data[i, :], peps_prime.data[i, :])
    neighbor_projectors = neighboring_tensors(SubNetwork(line_tensors), projectors)
    return tree(peps.data[i, :], peps_prime.data[i, :], neighbor_projectors)
  end
  subnetworks = [get_tree(i) for i in 1:Ny]
  return SubNetwork(subnetworks)
end

function inner_network(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  projectors::Vector{<:ITensor},
  mpo::MPO,
  coordinate::Tuple{<:Integer,Colon},
  ::typeof(tree),
)
  Ny, Nx = size(peps.data)
  function get_tree(i)
    if i == coordinate[1]
      line_tensors = vcat(peps.data[i, :], peps_prime_ham.data[i, :], mpo.data)
      neighbor_projectors = neighboring_tensors(SubNetwork(line_tensors), projectors)
      return tree(peps.data[i, :], peps_prime_ham.data[i, :], mpo.data, neighbor_projectors)
    else
      line_tensors = vcat(peps.data[i, :], peps_prime.data[i, :])
      neighbor_projectors = neighboring_tensors(SubNetwork(line_tensors), projectors)
      return tree(peps.data[i, :], peps_prime.data[i, :], neighbor_projectors)
    end
  end
  subnetworks = [get_tree(i) for i in 1:Ny]
  return SubNetwork(subnetworks)
end

function inner_network(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  projectors::Vector{<:ITensor},
  mpo::MPO,
  coordinate::Tuple{Colon,<:Integer},
  ::typeof(tree),
)
  Ny, Nx = size(peps.data)
  function get_tree(i)
    if i == coordinate[2]
      line_tensors = vcat(peps.data[:, i], peps_prime_ham.data[:, i], mpo.data)
      neighbor_projectors = neighboring_tensors(SubNetwork(line_tensors), projectors)
      return tree(peps.data[:, i], peps_prime_ham.data[:, i], mpo.data, neighbor_projectors)
    else
      line_tensors = vcat(peps.data[:, i], peps_prime.data[:, i])
      neighbor_projectors = neighboring_tensors(SubNetwork(line_tensors), projectors)
      return tree(peps.data[:, i], peps_prime.data[:, i], neighbor_projectors)
    end
  end
  subnetworks = [get_tree(i) for i in 1:Nx]
  return SubNetwork(subnetworks)
end

# Get the tensor network of <peps|mpo|peps'>
# The local MPO specifies the 2-site term of the Hamiltonian
function inner_network(
  peps::PEPS, peps_prime::PEPS, peps_prime_ham::PEPS, mpo::MPO, coordinates::Array
)
  @assert(length(mpo) == length(coordinates))
  network = vcat(peps.data...)
  dimy, dimx = size(peps.data)
  for ii in 1:dimx
    for jj in 1:dimy
      if (jj, ii) in coordinates
        index = findall(x -> x == (jj, ii), coordinates)
        @assert(length(index) == 1)
        network = vcat(network, [mpo.data[index[1]]])
        network = vcat(network, [peps_prime_ham.data[jj, ii]])
      else
        network = vcat(network, [peps_prime.data[jj, ii]])
      end
    end
  end
  return network
end

function flatten(v::Array{<:PEPS})
  tensor_list = [vcat(peps.data...) for peps in v]
  return vcat(tensor_list...)
end

function insert_projectors(peps::PEPS, cutoff=1e-15, maxdim=100)
  psi_bra = addtags(linkinds, dag.(peps.data), "bra")
  psi_ket = addtags(linkinds, peps.data, "ket")
  tn = psi_bra .* psi_ket
  bmps = boundary_mps(tn; cutoff=cutoff, maxdim=maxdim)
  psi_bra_rot = addtags(linkinds, dag.(peps.data), "brarot")
  psi_ket_rot = addtags(linkinds, peps.data, "ketrot")
  tn_rot = psi_bra_rot .* psi_ket_rot
  bmps = boundary_mps(tn; cutoff=cutoff, maxdim=maxdim)
  bmps_rot = boundary_mps(tn_rot; cutoff=cutoff, maxdim=maxdim)
  # get the projector for each center
  Ny, Nx = size(peps.data)
  bonds_row = [(i, :) for i in 1:Ny]
  bonds_column = [(:, i) for i in 1:Nx]
  tn_split_row, tn_split_column = [], []
  projectors_row, projectors_column = Vector{Vector{ITensor}}(), Vector{Vector{ITensor}}()
  for bond in bonds_row
    tn_split, pl, pr = insert_projectors(tn, bmps; center=bond)
    push!(tn_split_row, tn_split)
    push!(projectors_row, vcat(reduce(vcat, pl), reduce(vcat, pr)))
  end
  for bond in bonds_column
    tn_split, pl, pr = insert_projectors(tn_rot, bmps_rot; center=bond)
    push!(tn_split_column, tn_split)
    push!(projectors_column, vcat(reduce(vcat, pl), reduce(vcat, pr)))
  end
  return tn_split_row, tn_split_column, projectors_row, projectors_column
end

function insert_projectors(peps::PEPS, center::Tuple, cutoff=1e-15, maxdim=100)
  # Square the tensor network
  psi_bra = addtags(linkinds, dag.(peps.data), "bra")
  psi_ket = addtags(linkinds, peps.data, "ket")
  tn = psi_bra .* psi_ket
  bmps = boundary_mps(tn; cutoff=cutoff, maxdim=maxdim)
  tn_split, pl, pr = insert_projectors(tn, bmps; center=center)
  return tn_split, vcat(reduce(vcat, pl), reduce(vcat, pr))
end

"""Generate an array of networks representing inner products, <p|H_1|p>, ..., <p|H_n|p>
Parameters
----------
peps: a peps network with datatype PEPS
peps_prime: prime of peps used for inner products
peps_prime_ham: prime of peps used for calculating expectation values
Hs: An array of MPO operators with datatype LocalMPO
Returns
-------
An array of networks.
"""
function inner_networks(peps::PEPS, peps_prime::PEPS, peps_prime_ham::PEPS, Hs::Array)
  network_list = Vector{Vector{ITensor}}()
  for H_term in Hs
    if H_term isa Models.LocalMPO
      coords = [H_term.coord1, H_term.coord2]
    elseif H_term isa Models.LineMPO
      if H_term.coord[1] isa Colon
        coords = [(i, H_term.coord[2]) for i in 1:length(H_term.mpo)]
      else
        coords = [(H_term.coord[1], i) for i in 1:length(H_term.mpo)]
      end
    end
    inner = inner_network(peps, peps_prime, peps_prime_ham, H_term.mpo, coords)
    network_list = vcat(network_list, [inner])
  end
  return network_list
end

function inner_networks(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  projectors::Vector{<:ITensor},
  Hs::Array,
)
  network_list = inner_networks(peps, peps_prime, peps_prime_ham, Hs)
  return map(network -> vcat(network, projectors), network_list)
end

function inner_networks(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  projectors::Vector{Vector{ITensor}},
  Hs::Vector{Models.LineMPO},
)
  @assert length(projectors) == length(Hs)
  function generate_each_network(projector, H)
    return inner_networks(peps, peps_prime, peps_prime_ham, projector, [H])[1]
  end
  return [generate_each_network(projector, H) for (projector, H) in zip(projectors, Hs)]
end

function inner_networks(
  peps::PEPS,
  peps_prime::PEPS,
  peps_prime_ham::PEPS,
  projectors::Vector{Vector{ITensor}},
  Hs::Vector{Models.LineMPO},
  ::typeof(tree),
)
  @assert length(projectors) == length(Hs)
  function generate_each_network(projector, H)
    return inner_network(peps, peps_prime, peps_prime_ham, projector, H.mpo, H.coord, tree)
  end
  return [generate_each_network(projector, H) for (projector, H) in zip(projectors, Hs)]
end

function rayleigh_quotient(inners::Array)
  self_inner = inners[end][]
  expectations = sum(inners[1:(end - 1)])[]
  return expectations / self_inner
end