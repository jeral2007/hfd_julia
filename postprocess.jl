module HfdPostProcess
using LinearAlgebra
"average values of r powers in a.u./Z"
function moments(cpars, grid, occ_block, maxpow=2)
    res = zeros(eltype(grid.xs),length(occ_block.ks), maxpow+1)
    an = cpars.alpha*cpars.Z
    for pow=0:maxpow
        aux = grid.ws.*grid.xs.^(pow)
        for ki=1:length(occ_block.ks)
            gam = sqrt(occ_block.ks[ki]^2-an^2)
            pq = reshape(occ_block.vecs[:, ki], :, 2)
            res[ki, pow+1] = dot(aux, grid.xs .^ (2gam) .*(pq[:,1].^2 .+ pq[:,2].^2 .* an^2))/cpars.Z^pow
        end
    end
    res
end

"out table for occupied states"
function report(cpars, grid, occ_block)
    perm = sortperm((abs.(occ_block.ks) .+ occ_block.inds-(1 .+sign.(occ_block.ks))./2)*100+(abs.(occ_block.ks)+sign.(occ_block.ks)))
    [occ_block.ks[perm] occ_block.inds[perm] occ_block.ens[perm] moments(cpars, grid, occ_block)[perm, :]]
end
end
