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

function make_occ_blocks!(cpars, grid, occ_block; driver, N=250, kws...)
    avros = HfdPostProcess.moments(cpars, grid, occ_block)[:, 2]
    klusts = klust(avros)
    println("========================================")
    println("meta iterator with multigrid")
    println("occupied states grouped by <r>")
    for klust in klusts
        rep = [occ_block.ks occ_block.inds occ_block.ens avros][klust, :]
        display(rep)
    end
    occ_blocks = typeof(occ_block)[]
    grids = typeof(grid)[]
    cparss = typeof(cpars)[]
    for ii=1:length(klusts)
        klust = klusts[ii]
        println("processing cluster $ii")
        new_cpars = CalcParams(cpars.Z, N, cpars.alpha)
        new_grid = leg_rat_grid(N, 2*avros[klust[1]]*cpars.Z)
        new_block = sh_blk_new_grid(cpars, grid, new_grid, occ_block)
        driver(new_cpars, new_grid, new_block; active=klust, kws...)
        occ_block = sh_blk_new_grid(new_cpars, new_grid, grid, new_block)
        push!(cparss, new_cpars)
        push!(grids, new_grid)
        push!(occ_blocks, new_block)
    end
    cparss, grids, occ_blocks
end 

function klust(arr, tol = 12)
    inds = sortperm(arr)
    klusters = Vector{Int64}[[1]]
    for kk=2:length(inds)
      #  @show  dist(arr[inds[kk-1]], arr[inds[kk]])
        if arr[inds[kk]]<tol*arr[klusters[end][1]]
            push!(klusters[end], inds[kk])
        else
            push!(klusters, [inds[kk]])
        end
    end
    klusters
end   
