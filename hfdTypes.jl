module HfdTypes
using LinearAlgebra
include("polynodes.jl")
export CalcParams, Grid, leg_rat_grid, ShellBlock, from_dict, leg_exp_grid, project

"""struct to store calculation parameters
- Z::Real -- nuclear charge
- N::Int -- number of points of grid
- alpha::Real -- fine structure constant value (reciprocal to speed of light in atomic units)
"""
struct CalcParams{T}
    Z :: T
    N :: Int64
    alpha :: T
end

CalcParams(Z::T, N::Int64) where {T} = CalcParams{T}(Z, N, 0.0072973525643) # α value from wiki

abstract type Grid{T} end

struct RatGrid{T} <: Grid{T}
    xs:: Vector{T}
    ws:: Vector{T}
    dmat :: Matrix{T}
    ts:: Vector{T}
    g::QRPivoted{T, Matrix{T}, Vector{T}, Vector{Int64}}
    ker:: Matrix{T}
end

struct ExpGrid{T} <: Grid{T}
    xs:: Vector{T}
    ws:: Vector{T}
    dmat :: Matrix{T}
    ts:: Vector{T}
    g::QRPivoted{T, Matrix{T}, Vector{T}, Vector{Int64}}
    ker:: Matrix{T}
end
"constructs legendre rational functions grid"
function leg_rat_grid(N, k=2e2, rcut=Inf)
    as, gs = PolyNodes.alphas_leg(N), PolyNodes.gammas_leg(N)
    ts, ws = PolyNodes.nodes(as, gs, 2e0) 
    if rcut != Inf
        tcut = (rcut-k)/(rcut+k)
    else
        tcut = 1e0
    end
    mask =ts .<tcut
    ts = ts[mask]; ws = ws[mask]
    dmat = PolyNodes.diffOp(ts)
    xs, wxs = PolyNodes.caley_trans((ts, ws), k)
    dmat .*= (1 .-ts).^2 ./(2*k)
    dmat2 = dmat*dmat
    RatGrid(xs, wxs, dmat, ts, qr(dmat2,ColumnNorm()), nullspace(dmat2))
end
"constructs exponential grid"
function leg_exp_grid(N, x1)
    as, gs = PolyNodes.alphas_leg(N), PolyNodes.gammas_leg(N)
    ts, ws = PolyNodes.nodes(as, gs, 2e0)
    ct = log(x1-1)/(ts[end]+1)
    xs = exp.(ct.*(ts .+ 1e0)) .-1
    dxs = ct.*(xs .+ 1e0)
    wxs = ws .* dxs
    dmat = PolyNodes.diffOp(ts)
    @views dmat ./= dxs
    dmat2 = dmat*dmat
    ExpGrid(xs, wxs, dmat, ts, qr(dmat2,ColumnNorm()), nullspace(dmat2))
end    

mutable struct ShellBlock{T}
    ks :: Vector{Int64}
    occs :: Vector{T}
    inds :: Vector{Int64}
    vecs :: Matrix{T}
    ens  :: Vector{T}
end 


"""takes Dict(kappa => occ_arr) and returns
following arrays:
- ks - kappa values associated with each shell
- occs - occupation number for each shell
= inds - index of corresponding eigenvector (starts from 1, states with energy <-mec^2 are filtered)

also returns total nuclear charge ztot. ks, occs, inds
packed to the ShellBlock struct

Usage:
sh_block, ztot =  from_dict(occ_dict, Float64, N)

"""
function from_dict(occ, T, nx)
    kappas = Int64[]
    occs = T[]
    finds = Int64[]
    ztot = 0e0
    for (k, aux) in occ
        ind = 1
        for n_el in aux
            append!(kappas, k)
            append!(occs, n_el)
            append!(finds, k<0 ? ind : ind+1) #skip state with -1 nodes in k>0 case.
            if n_el>1e-5
                ztot += 2*abs(k)
            end
            ind += 1
        end
    end
    perm = sortperm(abs.(kappas))
    N = length(kappas)
    ShellBlock(kappas[perm], occs[perm], finds[perm], zeros(T, 2nx, N), 
               zeros(T, N)), ztot
end

function project(rat_grid::RatGrid, grid, func)
   inc(x) = x + one(x)
   k = rat_grid.xs[2]*inc(-rat_grid.ts[2])/inc(rat_grid.ts[2])
   tvals = (grid.xs .- k) ./ (grid.xs .+ k)
   func.(tvals)
end
function project(exp_grid::ExpGrid, grid, func)
   inc(x) = x + one(x)
   dec(x) = x - one(x)  
   res = similar(grid.ts) 
   c = log(inc(exp_grid.xs[2]))/inc(exp_grid.ts[2])
   for ii=1:length(res)
       tval= dec(log(inc(grid.xs[ii]))/c)
       if (tval<1) 
           res[ii] = func(tval)
       else
           res[ii] = 0e0
       end
   end
   res
end

end
