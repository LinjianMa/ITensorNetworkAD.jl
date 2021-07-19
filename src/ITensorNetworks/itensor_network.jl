
#
# Some general tools for working with networks of ITensors.
#

"""
    itensor_network(dims::Int...; linkdims)

Create a tensor network on a hypercubic lattice of
dimension `dims` with link dimension `linkdims`.

The network will have periodic boundary conditions.
To remove the periodic boundary condiitions, use
the function `project_boundary`.
"""
function itensor_network(dims::Int...; linkdims)
  return ITensor.(inds_network(dims...; linkdims=linkdims))
end

"""
    boundary_projectors(tn::Matrix{ITensor}, state=1)

For a 2D tensor network, return the right and bottom boundary projectors onto
the local state `state`.
"""
function boundary_projectors(tn::Matrix{ITensor}, state=1)
  top_row = tn[1, :]
  bottom_row = tn[end, :]
  left_column = tn[:, 1]
  right_column = tn[:, end]
  bottom_boundary_inds = commonind.(bottom_row, top_row)
  right_boundary_inds = commonind.(right_column, left_column)
  ψr = ITensors.state.(right_boundary_inds, state)
  ψb = ITensors.state.(bottom_boundary_inds, state)
  return ψr, ψb
end

"""
    project_boundary(tn::Matrix{ITensor}, state=1)

Project the boundary of a periodic 2D tensor network onto 
the specified state.
"""
function project_boundary(tn::Matrix{ITensor}, state=1)
  Nx, Ny = size(tn)
  ψr, ψb = boundary_projectors(tn, state)
  for n in 1:Nx
    tn[n, 1] = tn[n, 1] * ψr[n]
    tn[n, end] = tn[n, end] * dag(ψr[n])
  end
  for n in 1:Ny
    tn[1, n] = tn[1, n] * ψb[n]
    tn[end, n] = tn[end, n] * dag(ψb[n])
  end
  return tn
end

function filter_alllinkinds(f, tn)
  linkinds = Dict{Tuple{keytype(tn),keytype(tn)},Vector{indtype(tn)}}()
  for n in keys(tn), m in keys(tn)
    if f(n, m)
      is = commoninds(tn[n], tn[m])
      if !isempty(is)
        linkinds[(n, m)] = is
      end
    end
  end
  return linkinds
end

"""
    alllinkinds(tn)

Return a dictionary of all of the link indices of the network.
The link indices are determined by searching through the network
for tensors with indices in common with other tensors, and
the keys of the dictionary store a tuple of the sites with
the common indices.

Notice that this version will return a dictionary containing
repeated link indices, since the 
For example:
```julia
i, j, k, l = Index.((2, 2, 2, 2))
inds_network = [(i, dag(j)), (j, dag(k)), (k, dag(l)), (l, dag(i))]
tn_network = randomITensor.(inds_network)
links = allinkinds(tn_network)
links[(1, 2)] == (dag(j),)
links[(2, 1)] == (j,)
links[(2, 3)] == (dag(l),)
links[(1, 3)] # Error! In the future this may return an empty Tuple
```

Use `inlinkinds` and `outlinkinds` to return dictionaries without
repeats (such as only the link `(2, 1)` and not `(1, 2)` or vice versa).
"""
alllinkinds(tn) = filter_alllinkinds(≠, tn)
inlinkinds(tn) = filter_alllinkinds(>, tn)
outlinkinds(tn) = filter_alllinkinds(<, tn)

function filterneighbors(f, tn, n)
  neighbors_tn = keytype(tn)[]
  tnₙ = tn[n]
  for m in keys(tn)
    if f(n, m) && hascommoninds(tnₙ, tn[m])
      push!(neighbors_tn, m)
    end
  end
  return neighbors_tn
end

"""
    neighbors(tn, n)

From a tensor network `tn` and a site/node `n`, determine the neighbors
of the specified tensor `tn[n]` by searching for which other
tensors in the network have indices in common with `tn[n]`.

Use `inneighbors` and `outneighbors` for directed versions.
"""
neighbors(tn, n) = filterneighbors(≠, tn, n)
inneighbors(tn, n) = filterneighbors(>, tn, n)
outneighbors(tn, n) = filterneighbors(<, tn, n)

function mapinds(f, ::typeof(linkinds), tn)
  tn′ = copy(tn)
  for n in keys(tn)
    for nn in neighbors(tn, n)
      commonindsₙ = commoninds(tn[n], tn[nn])
      tn′[n] = replaceinds(tn′[n], commonindsₙ => f(commonindsₙ))
    end
  end
  return tn′
end

function ITensors.prime(::typeof(linkinds), tn, args...)
  return mapinds(x -> prime(x, args...), linkinds, tn)
end
function ITensors.addtags(::typeof(linkinds), tn, args...)
  return mapinds(x -> addtags(x, args...), linkinds, tn)
end

# Compute the sets of combiners that combine the link indices
# of the tensor network so that neighboring tensors only
# share a single larger index.
# Return a dictionary from a site to a combiner.
function combiners(::typeof(linkinds), tn)
  Cs = Dict(keys(tn) .=> (ITensor[] for _ in keys(tn)))
  for n in keys(tn)
    for nn in inneighbors(tn, n)
      commonindsₙ = commoninds(tn[n], tn[nn])
      C = combiner(commonindsₙ)
      push!(Cs[n], C)
      push!(Cs[nn], dag(C))
    end
  end
  return Cs
end

# Insert the gauge tensors `gauge` into the links of the tensor
# network `tn`.
function insert_gauge(tn, gauge)
  tn′ = copy(tn)
  for n in keys(gauge)
    for g in gauge[n]
      if hascommoninds(tn′[n], g)
        tn′[n] *= g
      end
    end
  end
  return tn′
end

# Insert the gauge tensors `gauge` into the links of the sets
# of tensor networks `tn` stored in a NamedTuple.
# TODO: is this used anywhere?
function insert_gauge(tn::NamedTuple, gauge)
  return map(x -> insert_gauge.(x, (gauge,)), tn)
end

# Split the links of an ITensor network by changing the prime levels
# or tags of pairs of links.
function split_links(H::Union{MPS,MPO}; split_tags=("" => ""), split_plevs=(0 => 1))
  left_tags, right_tags = split_tags
  left_plev, right_plev = split_plevs
  l = outlinkinds(H)
  Hsplit = copy(H)
  for bond in keys(l)
    n1, n2 = bond
    lₙ = l[bond]
    left_l_n = setprime(addtags(lₙ, left_tags), left_plev)
    right_l_n = setprime(addtags(lₙ, right_tags), right_plev)
    Hsplit[n1] = replaceinds(Hsplit[n1], lₙ => left_l_n)
    Hsplit[n2] = replaceinds(Hsplit[n2], lₙ => right_l_n)
  end
  return Hsplit
end

function split_links(H::Vector{ITensor}, args...; kwargs...)
  return data(split_links(MPS(H), args...; kwargs...))
end
