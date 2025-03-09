function readgraphdict(graph_stream)
  res = Dict()
  for line in eachline(graph_stream)
    (i, j, c) = split(chomp(line), ';')[1:3]
  println(i,j,c,'$')
  res[(i,j)] = float(c)
  end
  return res
end
Set
function graphmat_from_dict(dict)
vertexes = Set()
  for (i, j) in keys(dict)
    append!(vertexes
cd("/home/jeral/julia-play")
readgraphdict(open("graph.tsv","r"))

