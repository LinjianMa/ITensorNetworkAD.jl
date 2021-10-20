using ITensorNetworkAD
using AutoHOOT, ITensors, Zygote
using ITensorNetworkAD.ITensorNetworks:
  TreeTensor, uncontract_inds_binary_tree, tree_approximation

const itensorah = ITensorNetworkAD.ITensorAutoHOOT

@testset "test TreeTensor" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(2, "k")
  l = Index(4, "l")
  m = Index(5, "m")

  A = randomITensor(i, j, k)
  B = randomITensor(k, l, m)
  C = randomITensor(i, j, l, m)
  tree_A = TreeTensor(A)
  tree_B = TreeTensor(B)
  tree_C = TreeTensor(C)

  out = A * B
  network = [tree_A, tree_B]
  nodes, dict = itensorah.generate_einsum_expr([network])
  out_list = itensorah.compute_graph(nodes, dict; cutoff=1e-15, maxdim=1000)
  @test isapprox(out, ITensor(out_list[1]))

  out = A * B * C
  out2 = contract(tree_A, tree_B, tree_C; cutoff=1e-15, maxdim=1000)
  @test isapprox(out, ITensor(out2))
end

@testset "test batch_tensor_contraction" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(2, "k")
  A = randomITensor(i, j)
  B = randomITensor(j, k)
  C = randomITensor(k, i)

  function network(A)
    tree_A = TreeTensor(A)
    tree_B = TreeTensor(B)
    tree_C = TreeTensor(C)
    tensor_network = [tree_A, tree_B, tree_C]
    out = itensorah.batch_tensor_contraction(
      [tensor_network], tree_A; cutoff=1e-15, maxdim=1000
    )
    return sum(out)[]
  end
  grad_A = gradient(network, A)
  @test isapprox(grad_A[1], B * C)
end

@testset "test uncontract_inds_binary_tree" begin
  i = Index(2, "i")
  j = Index(2, "j")
  k = Index(2, "k")
  l = Index(2, "l")
  m = Index(2, "m")
  A = randomITensor(i)
  B = randomITensor(j)
  C = randomITensor(k)
  D = randomITensor(l)
  E = randomITensor(m)

  path = [[[A, B], [C, D]], E]
  uncontract_inds = [i, j, k, l, m]
  btree = uncontract_inds_binary_tree(path, uncontract_inds)
  @test btree == [[[[i], [j]], [[k], [l]]], [m]]
  out = tree_approximation([A, B, C, D, E], btree)
  @test isapprox(contract(out), A * B * C * D * E)
end
