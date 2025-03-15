module RadInts
include("3jsymb.jl")
"""evaluates radial integral of dens * r<^k/r^k+1. modifies density
result is multiplied by r"""
function radint!(cpars, grid, dens, k, res; green_funcs, j1=nothing, j2= nothing)
    if j1 != nothing
        fact2=symb3j0.gam2s(j1, j2, 2k)
    else
        fact2 = 1e0
    end
    @views res .= (green_funcs[:, :, k+1] * dens) .*grid.xs .* (fact2 / cpars.Z)
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

function excints!(cpars, grid, st, kappa1, occ_block, res; green_funcs)
    an2  = (cpars.alpha^2*cpars.Z^2) #kukuruznik
    lj(κ) = abs(κ) - Int((-sign(κ)+1)/2), 2*abs(κ)-1
    l1, j1 = lj(kappa1)
    γ1 = sqrt(kappa1^2 - an2)
    κs = sort(unique(occ_block.ks))
    pqs_occ = reshape(occ_block.vecs, cpars.N, 2, :)
    resPQ = reshape(res, :, 2)
    dens = zeros(eltype(grid.xs), cpars.N)
    excpot = zeros(eltype(grid.xs), cpars.N)
    for κ in κs
        γ2 = sqrt(κ^2 - an2)
        l2, j2 = lj(κ)
        f_inds = findall(occ_block.ks .== κ)
        kmin, kmax = abs(j1-j2), j1 + j2
        for f_i in f_inds
            dens .= 0e0
            excpot .= 0e0
            @views add_dens!(cpars, grid, st, occ_block.vecs[:, f_i], kappa1, κ,
                             occ_block.occs[f_i], dens)
            for kj=kmin:2:kmax
                if (l1+l2+Int(kj/2)) % 2 !=0
                    continue
                end
                radint!(cpars, grid, dens, Int(kj/2), excpot; green_funcs=green_funcs, j1=j1, j2=j2)
            end
            for ii =1:length(excpot)
                excpot[ii] *= grid.xs[ii]^(γ2-γ1)
                resPQ[ii, 1] -= excpot[ii] * pqs_occ[ii, 1, f_i]
                resPQ[ii, 2] -= excpot[ii] * pqs_occ[ii, 2, f_i] * an2
            end
        end
    end
end

function twoelint!(cpars, grid, st, kappa1, occ_block, res; cpot, green_funcs)
    an2  = (cpars.alpha^2*cpars.Z^2) #kukuruznik
    resPQ = reshape(res, :, 2)
    pq = reshape(st, :, 2)
    pqs_occ = reshape(occ_block.vecs, cpars.N, 2, :)
    @views begin
        resPQ[:, 1] .= cpot .* pq[:, 1]
        resPQ[:, 2] .= an2 .* cpot .* pq[:, 2]
    end
    excints!(cpars, grid, st, kappa1, occ_block, res; green_funcs=green_funcs)
end

function coulpot_func(cpars, grid, occ_block; green_funcs)
    dens = zeros(eltype(grid.xs), cpars.N)
    res = zeros(eltype(grid.xs), cpars.N)
    for f_i=1:length(occ_block.ks)
        @views add_dens!(cpars, grid, occ_block.vecs[:, f_i], occ_block.vecs[:, f_i], 
                         occ_block.ks[f_i], occ_block.ks[f_i],
                         occ_block.occs[f_i], dens)
    end
    radint!(cpars, grid, dens, 0, res; green_funcs = green_funcs)
end
end

module InvIter
using LinearAlgebra
using ..RadInts
using ..hfd_funcs

struct exc_func!{T}
    cpars:: hfd_funcs.Params{T}
    grid :: hfd_funcs.Grid{T}
    kappa :: Int64
    occ_block :: hfd_funcs.ShellBlock{T}
    green_funcs :: Array{T, 3}
end
function (a::exc_func!)(y, res)
    res .= zero(eltype(res))
    RadInts.excints!(a.cpars, a.grid, y, a.kappa, 
    a.occ_block, res; green_funcs=a.green_funcs)
end

function resolvent_iter!(H0, V, rhs, y, mat, u0, δu, v; dump=0.5)
    mul!(u0, H0, y)
    #v .= rhs*y
    mul!(v, rhs, y)
    V(y, δu)
    λ = (dot(y, u0)+dot(y, δu))/dot(y, v)
    @views u0 .+= -λ .* v .+ δu
    for ind in eachindex(mat)
        mat[ind] = H0[ind] - λ*rhs[ind]
    end
    @views u0 .*= dump
    @views y .-= qr!(mat, ColumnNorm()) \ u0
    @views y ./= norm(y)
    return λ, norm(u0)
end

function hfd_iter(cpars, grid, f_i, occ_block::hfd_funcs.HfdTypes.ShellBlock, cpot::Vector, green_funcs::Array; niter=10, dump=0.5, ntol=1e-6)
    exc! = exc_func!(cpars, grid, occ_block.ks[f_i], occ_block, green_funcs)

    y =( zeros(eltype(grid.xs),2cpars.N) .=occ_block.vecs[:, f_i])
    y0 = similar(y)
    u0 = similar(y)
    du = similar(y)
    r = similar(y)
    lhs = zeros(eltype(grid.xs), 2cpars.N, 2cpars.N)
    rhs = similar(lhs)
    hfd_funcs.dirac_h1!(cpars, grid, occ_block.ks[f_i], lhs, rhs)
    mat = similar(lhs)
    an2 = (cpars.alpha*cpars.Z)^2 # guess what
    for ii=1:cpars.N
        lhs[ii, ii] += cpot[ii]
        lhs[cpars.N+ii, cpars.N + ii] += cpot[ii] * an2
    end
    
    function process_shell!(y, y0; dump=dump, niter=niter)
        copy!(y0, y)
        en0, rn = resolvent_iter!(lhs, exc!, rhs, y, mat, u0, du, r;dump=dump)
        for iter=1:niter
            en, rn = resolvent_iter!(lhs, exc!, rhs, y, mat, u0, du, r;dump=dump)
            @show en * cpars.Z^2, rn, iter
            if (rn .<ntol) 
                return en, true
            end
            if en.>0
                @warn "not converged -- positive energy, try to continue with dump factor /4"
                copy!(y, y0)
                return en, false
            end
            en_rel = 2abs(en-en0)/(en+en0)
            if en_rel>1
                @warn "eigen value changed too much: en_rel=$en_rel; restart with dump factor/4"
                copy!(y, y0)
            end
        end
        @warn "convergence not achieved after $niter iterations"
        return en, false
    end
    en, conv = process_shell!(y, y0; dump=dump)
    if (!conv)
        en, conv = process_shell!(y, y0; dump=dump/4, niter=4*niter)
    end
    en
end
end
