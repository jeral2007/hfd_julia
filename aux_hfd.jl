module hfd_funcs
include("hfdTypes.jl")
include("3jsymb.jl")
include("matutils.jl")
include("postprocess.jl")
using LinearAlgebra
using .HfdTypes

"""dirac one-electron hamiltonian for radial functions
- cpars::CalcParams -- calculation parameteres 
- grid::Grid -- radial grid nodes, weights etc
- kappa - angular quantum number

OUTPUT: lhs, rhs - matrices for generalized eigen value problem
eigen vectors for this problem are of form [P; Q] where 
r^gamma*P -- big component values on grid,  
r^gamma*Q*Z/c -- small component values
"""
function dirac_h1(cpars, grid, kappa)
    an = cpars.alpha*cpars.Z
    sc_fact = cpars.scale*cpars.Z
    gamma = sqrt(kappa^2- an^2)
    eye = diagm(ones(length(grid.xs)))
    δnuc_mat = diagm(δnuc(cpars, grid))
    xmat = diagm(grid.xs)
    lhsPP = -eye - δnuc_mat
    lhsPQ = (gamma-kappa) .* eye .+ xmat*grid.dmat
    lhsQQ = -2 .* xmat.*sc_fact .- (eye + δnuc_mat).*an^2
    lhsQP = -(gamma+kappa) .* eye .- xmat*grid.dmat
    lhs = [lhsPP lhsPQ;
           lhsQP lhsQQ]
    rhsP = xmat
    zm = zeros(length(grid.xs), length(grid.xs))
    rhs = [rhsP zm;
           zm rhsP.*an^2]
    (lhs.*sc_fact, rhs)
end

function dirac_h1!(cpars, grid, kappa, lhs, rhs)
    an = cpars.alpha*cpars.Z
    gamma = sqrt(kappa^2- an^2)
    N = length(grid.xs)
    eye = diagm(ones(eltype(grid.xs), N))
    xmat = diagm(grid.xs)
    dnucvals = δnuc(cpars, grid)
    sc_fac = cpars.scale*cpars.Z
    @views begin
    lhsPP = lhs[1:N, 1:N]
    lhsPQ = lhs[1:N, N+1:end]
    lhsQP = lhs[N+1:end,1:N]
    lhsQQ = lhs[N+1:end, N+1:end]
    lhsPP .= -eye 
    lhsPQ .= (gamma-kappa) .* eye .+ xmat*grid.dmat
    lhsQQ .= -2 .* xmat.*sc_fac .- eye.*an^2
    lhsQP .= -(gamma+kappa) .* eye .- xmat*grid.dmat
    end
    for kk=1:N
        lhs[kk, kk] += dnucvals[kk]
        lhs[kk+N, kk+N] += -an^2 *dnucvals[kk]
    	rhs[kk,kk] = grid.xs[kk]
    	rhs[kk+N, kk+N] = an.^2 * grid.xs[kk]
    end
    lhs .*= sc_fac
end

"""electrostatic potential on grid
- cpars::CalcParams -- calculation parameteres 
- grid::Grid -- radial grid nodes, weights etc
- kappa - angular quantum number
-pq - [P; Q] Array with P and Q values on grid

OUTPUT:
 Zel(r) = U(r)*r on grid, U(r) - electrostatic potential
"""
function coul(cpars, grid, kappa, pq)
    xs = grid.xs
    ws = grid.ws
    alpha = cpars.alpha
    Z = cpars.Z
    T = eltype(xs)
    N = cpars.N
    an = alpha*Z
    gamma = sqrt(kappa^2-an^2)
    dens = zeros(T,N)
    @views begin
        P = pq[1:N]
        Q = pq[N+1:end]
    end
    dens .= P .^2 + (an.*Q) .^2
    mask = abs.(dens).<1e-15
    dens .*= xs.^(2*gamma)
#    dens[mask] .= 0e0
    intl = zero(T)
    intg = dot(ws, dens./xs)
    res = zeros(T, N)
    @inbounds for ii=1:N
        intl += ws[ii]*dens[ii]
        intg -= ws[ii]*dens[ii]/xs[ii]
        res[ii] = (intl+xs[ii]*intg)
    end
    res
end


rcut(en) = (-log(eps(en))/sqrt(2*abs(en)))

"""normalize given [P Q] bispinor, Q is the small component divided by αZ
- kappa -- relativistic angular quantum number
- cpars::CalcParams -- calculation parameteres 
- grid::Grid -- radial grid nodes, weights etc
- pq -- [P Q] values on nodes
first half of pq is the big component and rest is the scaled small component
rcut - cutting radius
returns normalized pq vector
"""
function normalize!(cpars, grid, pq, kappa, rcut)
    an = cpars.alpha*cpars.Z
    gamma = sqrt(kappa^2-an^2)
    @views begin
        P = pq[1:cpars.N]
        Q = pq[cpars.N+1:end]
    end
    mask = grid.xs .< rcut
    dens = (P.^2 + an^2*Q.^2)
    dens .*= grid.xs.^(2*gamma)
    nrm2 = dot(grid.ws[mask], dens[mask])
    pq ./= sqrt(nrm2)
    P[.!mask] .= 0e0
    Q[.!mask] .= 0e0
    pq
end

"""returns electrostatic potential of occupied states
- cpars -- calculation parameters
- grid -- grid nodes and weights packed to _Grid_ struct
- occ_block -- data of occupied states packed to _ShellBlock_ struct
"""
#function coul_pot(cpars::CalcParams{T}, grid::Grid{T}, occ_block::ShellBlock{T}, kappa) where {T}
#    res = zeros(T, cpars.N)
#    an = cpars.alpha*cpars.Z
#    @inbounds for ii=1:length(occ_block.ks)
#            res .+= occ_block.occs[ii].*coul(cpars, grid, occ_block.ks[ii],
#                                                                      occ_block.vecs[:, ii])./cpars.Z
#    end
#    diagm([res; an^2 .* res])
#end
function coul_pot(cpars::Params{T}, grid::Grid{T}, occ_block::ShellBlock{T}, kappa) where {T}
    res = zeros(T, 2cpars.N)
    resmat = zeros(T, 2cpars.N, 2cpars.N)
    an = cpars.alpha*cpars.Z
    dens = zeros(T, cpars.N)
    pqs = reshape(occ_block.vecs, :, 2, length(occ_block.ks))
    @views for ii=1:length(occ_block.ks)
        γ = sqrt(occ_block.ks[ii]^2-an^2)
        dens .+= (pqs[:, 1, ii].^2 + an^2 .* pqs[:, 2, ii].^2) .* occ_block.occs[ii].*grid.xs.^(2γ)
    end
    res2c = reshape(res, :, 2)

    @views res2c[:, 1] .= grid.xs .* (grid.pot * dens).*cpars.scale
    @views res2c[:, 2] .= res2c[:, 1] .* an^2
    for ii=1:2cpars.N
        resmat[ii, ii] = res[ii]
    end
    resmat
end

"""saves hamiltonian and right hand part of dirac eq to preallocated buffers"""
function lhs_rhs!(cpars, grid::Grid, kappa::Int, occ_block::ShellBlock, lhs, rhs; ecp=nothing, pot_func)
    dirac_h1!(cpars, grid, kappa, lhs, rhs)
    @views lhs .+= pot_func(cpars, grid, occ_block, kappa)
    if ecp != nothing
        ecp_vals = zeros(eltype(grid.xs), cpars.N*2)
        @views ecp_vals[1:cpars.N] .= ecp(cpars, grid, kappa)
        @views ecp_vals[cpars.N+1:end] .= (cpars.alpha*cpars.Z).^2 .* ecp_vals[1:cpars.N]
        @views lhs .+= diagm(ecp_vals) 
    end
    lhs, rhs
end

"""perform calculation for occupied states
- cpars -- calculation parameters
- grid -- grid nodes and weights packed to _Grid_ struct
- occ_block -- data of occupied states packed to _ShellBlock_ struct

Keyword arguments:
- pot_func -- callback to calculate electron interaction potential with
signature _pot\\_func(cpars::CalcParams{T}, grid::Grid{T}, kappa)::Matrix{T}_. This function
returns matrix of size (2N, 2N) to describe potential acting on big and small components. The _N_
is the number of radial grid nodes.
- maxiter -- maximum number of iterations of self-consistent calculation
- tol -- if maximum difference between old and newpot is less than tol, the convergence is achieved.
- dump -- dumping parameter that controls mixing with previous iteration
- indexes of active orbitals in occ_block, if nothing -- all orbitals active.
- caption -- header to print at start of iterations
occ_block is updated as result of calculation
"""    
function calc_occ!(cpars::Params{T}, grid::Grid{T}, 
    occ_block::ShellBlock{T}; pot_func, 
    maxiter=50, tol::T = 1e-6, dump::T = 0.5, caption, ecp=nothing, aitken=false, active = nothing) where {T}
    κs = sort(unique(occ_block.ks))
    rhs = zeros(T, cpars.N*2, cpars.N*2)
    lhs = zeros(T, size(rhs))
    N = cpars.N
    hs = zeros(T, 2N, 2N, length(κs))
    old_hs = zeros(T, size(hs))
    if aitken
       old_hs2 = zeros(T, size(hs)) # previous previous iteration for aitken δ2 process
    end
    make_lhs_rhs(kappa, lhs, rhs) = lhs_rhs!(cpars, grid, kappa, occ_block, lhs, rhs; ecp, pot_func)
    function make_kmask(kappa)
        kmask = kappa .== occ_block.ks
        if active != nothing
            kmask .= kmask .&& in(active).(eachindex(occ_block.ks))
        end
        kmask
    end
    for (ii, kappa) in enumerate(κs)
        @views make_lhs_rhs(kappa, hs[:, :, ii], rhs)
    end
    header="""
+=========================================================
|Starting calc_occ iterations                             
|atomic charge: $(cpars.Z)
!atomic scale: $(cpars.scale) a.e
|fine structure constant: $(cpars.alpha)
|number of nodes in radial grid: $(cpars.N)
|maximal iteration number: $maxiter
|tolerance: $tol
|dumping: $dump 
|aitken accelerated: $aitken
|$caption\n"""
    if ecp!=nothing
       header = header*"""
semilocal ecp with $(ecp.Nel) core electrons, lmax=$(length(ecp.lblocks)-1), nso = $(length(ecp.lblocks))"""
    end 
header = header*"""\n
+========================================================="""
    println(header)
    maxdelta = zeros(T, size(κs))
    for iter=1:maxiter
        println("iter no: $iter")
        for ki=1:length(κs)
            kappa = κs[ki]
	    kmask = make_kmask(kappa)
            if all(.!kmask) 
		continue
            end
            if aitken
                old_hs2[:, :, ki] .= old_hs[:, :, ki]
            end
	    old_hs[:, :, ki] .= hs[:, :, ki]
            ens, aux = eigen(hs[:, :, ki], rhs)
            inds = findall(real(ens).>-(cpars.Z * cpars.scale)^2)[occ_block.inds[kmask]]
            oinds = findall(kmask)
            occ_block.ens[kmask] .= real(ens[inds])./cpars.scale^2
            for (fi, vi) in zip(oinds, inds)
                if (active!=nothing) && (fi in active)
                    continue
                end
                rc = min(rcut(real(ens[vi])), grid.xs[end])
                occ_block.vecs[:, fi] .= normalize!(cpars, grid, real(aux[:, vi]), kappa, rc)
            end
        end
        println("κ      |δpot|")
        for ki=1:length(κs)
            kappa = κs[ki]
	    kmask = make_kmask(kappa)
            if all(.!kmask) 
		continue
            end
            @views make_lhs_rhs(kappa, hs[:, :, ki], rhs)
	    delta = abs.(old_hs[:, :, ki] .- hs[:, :, ki])
	    maxind = argmax(delta)
	    mi, mj = [maxind[1], maxind[2]] .% N .+ 1
            #maxdelta[ki] = delta[maxind]*exp(-1e-3*(grid.xs[1+(maxind[1] % (N+1))]+grid.xs[1+(maxind[2] % (N+1))])) #to exclude inf points
            maxdelta[ki] = delta[maxind]/sqrt(grid.xs[mi]*grid.xs[mj])
            println("$kappa  $(maxdelta[ki])")
            @views hs[:, :, ki] .= hs[:, :, ki] .* dump .+ old_hs[:, :, ki].*(1-dump)
        end
 	    if aitken && (maxiter-2 > iter>2) && (iter  % 6 == 0)
		MatUtils.accelerate!(hs, old_hs, old_hs2)
            end  
        println("orbital energies:")
        println(occ_block.ens)
        println("=========================================================")
        if maximum(maxdelta)<tol
            break
        end
    end
    hs
end

hcore_pot(cpars, grid, occ_block, kappa)= zeros(eltype(cpars.Z), cpars.N*2, cpars.N*2)

hcore_calc!(cpars, grid, occ_block; kws...) = calc_occ!(cpars, grid, occ_block; 
                                                pot_func=hcore_pot, 
                                                maxiter=3, 
                                                caption="non-interacting electrons approximation", kws...)

function sc_coul(cpars, grid, occ_block, kappa) 
    ztot = sum(occ_block.occs)
    coul_pot(cpars, grid, occ_block, kappa) .* (1 - 1/ztot)
end

sc_coul_calc!(cpars, grid, occ_block; kws...) = calc_occ!(cpars, grid, occ_block; 
                                                          caption="scaled (Z-1/Z) coulumb interaction", 
                                                          pot_func=sc_coul,
                                                          kws...)
"""auxilary function to evaluate exchange matrix. 
Arguments:
- cpars::Params, grid::Grid -- Parameters of Calculation and grid.
- k1::Int -- κ value for Hamiltonian Hκ
- k2::Int -- κ value of electronic state 
- pq::Vector -- values of big and scaled small component of electronic state on grid.
- occ::Float -- occupation number  of shell of state
- res::Matrix -- at the end of exc_func! evaluation, exchange part
corresponding to interaction with state pq 
will be added to res"""
function exc_func!(cpars, grid, k1, k2, pq, occ, res)
    lj(κ) = abs(κ) - Int((-sign(κ)+1)/2), 2*abs(κ)-1
    l1, j1 = lj(k1)
    l2, j2 = lj(k2)
    an = cpars.alpha*cpars.Z
    gam1 = sqrt(k1^2-an^2)
    gam2 = sqrt(k2^2-an^2)
    kmin, kmax = abs(j1-j2), j1 + j2
    @inbounds for ii=1:cpars.N, jj=1:cpars.N
        #fact = grid.xs[jj]^(gam1+gam2)*grid.ws[jj]*occ
        #fact = grid.xs[jj]^(gam1+gam2)*grid.pot[ii, jj]*occ/cpars.Z
        fact = grid.xs[jj]^(gam1+gam2)*occ*grid.xs[ii]*cpars.scale
        if grid.xs[ii]>eps(eltype(grid.xs))
            fact*=grid.xs[ii]^(gam2-gam1)
        else
            fact = 0e0
        end
        rg, rl = max(grid.xs[ii], grid.xs[jj]), min(grid.xs[ii], grid.xs[jj])
        for kj=kmin:2:kmax
            pk = div(kj, 2)
            if (l1+l2+pk) % 2 !=0
                continue
            end
            #if pk == 0 # to ensure exact cancelation of coulumb and exchange for same state
             #   tmp = grid.pot[ii, jj] * grid.xs[jj]^(2gam1)*grid.xs[ii]*occ*cpars.scale
             #   tmp *= symb3j0.gam2s(j1, j2, 0)
             #   res[ii, jj] -= tmp*pq[ii]*pq[jj]
             #   res[cpars.N+ii, cpars.N+jj] -= tmp*pq[cpars.N+ii]*pq[cpars.N+jj]*an^2
             #   continue
            #end
            fact2=symb3j0.gam2s(j1, j2, kj)*grid.pots[ii, jj, pk+1]
            #fact2=symb3j0.gam2s(j1, j2, kj)*rl^pk/rg^(pk+1)
            res[ii, jj] -= fact*fact2*pq[ii]*pq[jj] #big component
            res[ii, jj+cpars.N] -= fact*fact2*pq[ii]*pq[jj+cpars.N]*an^2 #big component
            res[cpars.N+ii, cpars.N+jj] -=fact*fact2*an^4*pq[ii+cpars.N]*pq[jj+cpars.N] # small component
            res[cpars.N+ii, jj] -=fact*fact2*an^2*pq[ii+cpars.N]*pq[jj] # small component
        end
    end
end

"""exchange part of electron interaction"""
function exc_pot!(cpars, grid, occ_block, kappa, res)
    κ_vals = sort(unique(occ_block.ks))
    for k2 in κ_vals
        f_inds = findall(k2 .== occ_block.ks)
        for fi in f_inds
            exc_func!(cpars, grid, kappa, k2, 
                      occ_block.vecs[:, fi], occ_block.occs[fi], res)
        end
    end
    res
end
"""Hartree Fock mean field matrix of occupied shells in occ_block. Result for κ part of Hamiltonian"""
function hfd_pot(cpars, grid, occ_block, kappa)
    res = coul_pot(cpars, grid, occ_block, kappa)
    exc_pot!(cpars, grid, occ_block, kappa, res)
    res
end

"Hartree mean field + exchange with occupied shells with same κ (dominant exchange part, almost slater hole)"
function hded_pot(cpars, grid, occ_block, kappa)
    res = coul_pot(cpars, grid, occ_block, kappa)
    an = (cpars.alpha*cpars.Z)
    γ = sqrt(kappa^2-an2)
    for f_i=1:length(occ_block.ks)
        if (occ_block.ks[f_i] != kappa)
            continue
        end
        for jj=1:cpars.N, ii=1:cpars.N
            fact = occ_block.occs[f_i]/(2abs(occ_block.ks[f_i])) * grid.pot[ii, jj]*grid.xs[ii]*cpars.scale
            fact *= grid.xs[jj]^(2γ)
            res[ii, jj] -= fact*occ_block.vecs[ii, f_i] * occ_block.vecs[jj, f_i]
            res[ii, jj+ cpars.N] -= fact*occ_block.vecs[ii, f_i] * occ_block.vecs[jj+cpars.N, f_i]*an^2
            res[ii+cpars.N, jj] -= fact*occ_block.vecs[ii+cpars.N, f_i] * occ_block.vecs[jj, f_i]*an^2
            res[ii+cpars.N, jj+cpars.N] -= fact*occ_block.vecs[ii+cpars.N, f_i] * occ_block.vecs[jj+cpars.N, f_i]*an^4
        end
    end
    res
end
            
hfd_calc!(cpars, grid, occ_block; kws...) = calc_occ!(cpars, grid, occ_block; 
                                                          caption="Dirac Hartree Fock", 
                                                          pot_func=hfd_pot,
                                                          kws...)

hded_calc!(cpars, grid, occ_block; kws...) = calc_occ!(cpars, grid,occ_block;
                                                      caption="Dirac Hartree + dominant exchange part",
                                                      pot_func=hded_pot,
                                                      kws...)


include("refine_occ.jl")
end

