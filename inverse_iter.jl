module RadInts
include("3jsymb.jl")
"""evaluates radial integral of dens * r<^k/r^k+1. modifies density
result is multiplied by r"""
function radint!(cpars, grid, dens, k, res; pots, j1=nothing, j2= nothing)
    if j1 != nothing
        fact2=symb3j0.gam2s(j1, j2, 2k)
    else
        fact2 = 1e0
    end
    @views res .= (pots[:, :, k+1] * dens) .*grid.xs .* (fact2 / cpars.Z)
    res
end
  
function add_dens!(cpars, grid, st1, st2, kappa1, kappa2, occ, res)
  an = cpars.alpha*cpars.Z
  gam1 = sqrt(kappa1^2 - an^2)
  gam2 = sqrt(kappa2^2 - an^2)
  pq1 = reshape(st1, cpars.N, 2)
  pq2 = reshape(st2, cpars.N, 2)
  @views res .+= ((pq1[:, 1].*pq2[:, 1] .+ pq1[:, 2] .* pq2[:, 2] .* an^2) .*
                  grid.xs .^ (gam1+gam2)).* occ
end

function twoelint!(cpars, grid, st, kappa1, occ_block, res; cpot, pots)
    an2  = (cpars.alpha^2*cpars.Z^2) #kukuruznik
    lj(κ) = abs(κ) - Int((-sign(κ)+1)/2), 2*abs(κ)-1
    resPQ = reshape(res, :, 2)
    pq = reshape(st, :, 2)
    pqs_occ = reshape(occ_block.vecs, cpars.N, 2, :)
    @views begin
        resPQ[:, 1] .= cpot .* pq[:, 1]
        resPQ[:, 2] .= an2 .* cpot .* pq[:, 2]
    end
    l1, j1 = lj(kappa1)
    γ1 = sqrt(kappa1^2 - an2)
    κs = sort(unique(occ_block.ks))
    for κ in κs
        γ2 = sqrt(κ^2 - an2)
        l2, j2 = lj(κ)
        f_inds = findall(occ_block.ks .== κ)
        kmin, kmax = abs(j1-j2), j1 + j2
        for f_i in f_inds
            dens = zeros(eltype(grid.xs), cpars.N)
            excpot = zeros(eltype(grid.xs), cpars.N)
            @views add_dens!(cpars, grid, st, occ_block.vecs[:, f_i], kappa1, κ,
                             occ_block.occs[f_i], dens)
            for kj=kmin:2:kmax
                if (l1+l2+Int(kj/2)) % 2 !=0
                    continue
                end
                radint!(cpars, grid, dens, Int(kj/2), excpot; pots, j1=j1, j2=j2)
            end
            excpot .*= grid.xs.^(γ2-γ1)
            resPQ[:, 1] .-= excpot .* pqs_occ[:, 1, f_i]
            resPQ[:, 2] .-= excpot .* pqs_occ[:, 2, f_i] .* an2
        end
    end
end

function coulpot_func(cpars, grid, occ_block; pots)
    dens = zeros(eltype(grid.xs), cpars.N)
    res = zeros(eltype(grid.xs), cpars.N)
    for f_i=1:length(occ_block.ks)
        @views add_dens!(cpars, grid, occ_block.vecs[:, f_i], occ_block.vecs[:, f_i], 
                         occ_block.ks[f_i], occ_block.ks[f_i],
                         occ_block.occs[f_i], dens)
    end
    radint!(cpars, grid, dens, 0, res; pots)
end
end
