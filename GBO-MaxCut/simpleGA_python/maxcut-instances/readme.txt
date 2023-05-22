All edges have edge weights randomly generated following a Beta
distribution with parameters  alpha = 100,  beta = 1 and scaled to the
range of [1;5].

setA: fully connected graphs.
setB: 2D square-grid graphs.
setC: Geometric, randomly distributed in the 1000xunit box.
      Distances are floor(Euclidean). Edge from each vertex to its floor(sqrt(l)) nearest neighbors.
setD: Chain of cliques of size 5, connected by a single edge.
setE: Chain of cliques of size 5, connected by a single edge; problem indices permuted.
