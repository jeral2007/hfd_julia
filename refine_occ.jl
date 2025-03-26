function sh_blk_new_grid(cpars_old, old_grid, cpars_new, new_grid, occ_block)
    an2 = cpars_new.alpha*cpars_new.Z
    ks = occ_block.ks
    inds = occ_block.inds
    occs = occ_block.occs
    ens = occ_block.ens
    vecs  = zeros(eltype(new_grid.xs), 2*length(new_grid.xs), length(occ_block.ks))
    pqs = reshape(vecs, :, 2, length(occ_block.ks))
    old_pqs = reshape(occ_block.vecs, :, 2, length(occ_block.ks))
    for kk=1:length(ks), i_pq=1:2
        gam = sqrt(ks[kk]^2 - an2)
        func = HfdTypes.PolyNodes.Lagrange(old_grid.ts, old_pqs[:, i_pq, kk])
        @views pqs[:, i_pq, kk] .= HfdTypes.project(cpars_old, old_grid, cpars_new, new_grid, func)
        
        xmask = new_grid.xs .> rcut(ens[kk]*cpars_new.scale^2)
        pqs[xmask, i_pq, kk] .= 0e0
        @views pqs[:, i_pq, kk] .*= (cpars_new.scale/cpars_old.scale)^(2gam)
    end
    for kk=1:length(ks)
        normalize!(cpars_new, new_grid, vecs[:, kk], ks[kk], rcut(ens[kk]*cpars_new.scale^2))
    end
    HfdTypes.ShellBlock(ks, occs, inds, vecs, ens)
end

function make_occ_blocks!(cpars, grid, occ_block; N=57, driver, kws...)
    ravs = HfdPostProcess.moments(cpars, grid, occ_block, 1)[:, 2]
    perm = sortperm(ravs)
    klusts = Integer.(round.(log.(ravs./ravs[perm[1]])./log(3); digits=0))
    klust_nos = sort(unique(klusts))
    MatUtils.kmeans!(klusts, log.(ravs./ravs[perm[1]]))
    scales = map(klust_nos) do klust
        kis = findall(klusts .== klust) 
        mean = sum(log.(ravs[kis]/ravs[1])) / length(kis)
        ravs[1] .*exp(mean)
    end
    
    occ_blocks = typeof(occ_block)[]
    grids = typeof(grid)[]
    cparss = typeof(cpars)[]
    for ii=1:length(klust_nos)
        klust = klust_nos[ii]
        act_inds = findall(klusts .== klust)
        println("processing cluster $ii")
        new_cpars = CalcParams(cpars.Z, N;alpha=cpars.alpha, scale=scales[ii])
        new_grid = leg_rat_grid(N, 1)
        new_block = sh_blk_new_grid(cpars, grid, new_cpars, new_grid, occ_block)
        driver(new_cpars, new_grid, new_block; active=act_inds, kws...)
        occ_block = sh_blk_new_grid(new_cpars, new_grid, cpars, grid, new_block)
        push!(cparss, new_cpars)
        push!(grids, new_grid)
        push!(occ_blocks, new_block)
    end
    cparss, grids, occ_blocks
end 

