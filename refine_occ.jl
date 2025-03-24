function sh_blk_new_grid(cpars, old_grid, new_grid, occ_block)
    ks = occ_block.ks
    inds = occ_block.inds
    occs = occ_block.occs
    ens = occ_block.ens
    vecs  = zeros(eltype(new_grid.xs), 2*length(new_grid.xs), length(occ_block.ks))
    pqs = reshape(vecs, :, 2, length(occ_block.ks))
    old_pqs = reshape(occ_block.vecs, :, 2, length(occ_block.ks))
    for kk=1:length(ks), i_pq=1:2
        func = HfdTypes.PolyNodes.Lagrange(old_grid.ts, old_pqs[:, i_pq, kk])
        @views pqs[:, i_pq, kk] .= HfdTypes.project(old_grid, new_grid, func)
        xmask = new_grid.xs .> rcut(ens[kk]/cpars.Z^2)
        pqs[xmask, i_pq, kk] .= 0e0
    end
    HfdTypes.ShellBlock(ks, occs, inds, vecs, ens)
end

function make_occ_blocks!(cpars, grid, occ_block; N=57)
    ravs = HfdPostProcess.moments(cpars, grid, occ_block, 1)[:, 2]
    perm = sortperm(ravs)
    klusts = Integer.(round.(log.(ravs./ravs[perm[1]])./log(3); digits=0))
    MatUtils.kmeans!(klusts, log.(ravs./ravs[perm[1]]))
    scales = map(sort(unique(klusts))) do klust
        kis = findall(klusts .== klust) 
        mean = sum(log.(ravs[kis]/ravs[1])) / length(kis)
        ravs[1] .*exp(mean)
    end
    
    return scales
    occ_blocks = typeof(occ_block)[]
    grids = typeof(grid)[]
    cparss = typeof(cpars)[]
    for ii=1:length(klusts)
        klust = klusts[ii]
        println("processing cluster $ii")
        new_cpars = CalcParams(cpars.Z, N;alpha=cpars.alpha, scale=scales[ii])
        new_grid = leg_rat_grid(N, 1)
        new_block = sh_blk_new_grid(cpars, grid, new_grid, occ_block)
        driver(new_cpars, new_grid, new_block; active=klust, kws...)
        occ_block = sh_blk_new_grid(new_cpars, new_grid, grid, new_block)
        push!(cparss, new_cpars)
        push!(grids, new_grid)
        push!(occ_blocks, new_block)
    end
    cparss, grids, occ_blocks
end 

