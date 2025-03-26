module GausECP
export ECPL, @ecp_blk,  @_r
as::Vector{Float64} = []
ps::Vector{Int64} = []
cs::Vector{Float64} = []
perm::Vector{Int64} = [1, 2, 3]
function flush_block()
   global as, ps, cs
   as, ps, cs = [], [], []
end
function add_row(a, p, c)
   global as, ps, cs
   global perm
   a, p, c = [a, p, c][perm]
   push!(as, a)
   push!(ps, p)
   push!(cs, c)
end

macro _r(a, p, c)
   :(add_row($a, $p, $c))
end

macro ecp_blk(arr, code)
   quote
        global as, ps, cs
	flush_block()
	$code
        push!($(esc(arr)), ECPL(copy(as), copy(ps), copy(cs)))
   end
end
"""struct to store ecp lblock data
ecp lblock is the function ΔVl(r) of following form:
ΔVl(r) = Σc_i r^(p-2) exp(-αi*r^2)

fields of struct:
- **alphas:: Vector{T}** -- vector of exponent αs
- **pows :: Vector{T}** -- vector of ps values
- **coefs :: Vector{T}** -- vector of coefs ci
"""
struct ECPL{T}
    alphas :: Vector{T}
    pows   :: Vector{Int64}
    coefs  :: Vector{T}
end
end
module RECP
using ..GausECP
function lblock_on_grid(cpars, grid, lblock)
    res = zeros(eltype(grid.xs), length(grid.xs))
    for kk=1:length(lblock.alphas)
        α = lblock.alphas[kk]*cpars.scale^2
        res .+=grid.xs .^ (lblock.pows[kk]-1) .* (exp.(-α .* grid.xs .^ 2) .* 
                                                  (lblock.coefs[kk]*cpars.scale^(lblock.pows[kk])))
    end
    res
end

struct ECPnum
    Nel
    lblocks
    soblocks
end

ECPnum((N, lbs, sbs)) = ECPnum(N, lbs, sbs)
function (ecp::ECPnum)(cpars, grid, kappa)
    l = abs(kappa)+sign(kappa)
    j = abs(kappa)-1/2
    l = Int(j + sign(kappa)/2)
    s = -sign(kappa)
    res = ecp.Nel*cpars.scale .*ones(eltype(grid.xs), length(grid.xs))
    res .+=lblock_on_grid(cpars, grid, ecp.lblocks[1])
    if (l+2)<=length(ecp.lblocks)
        res .+=lblock_on_grid(cpars, grid, ecp.lblocks[l+2])
    end
    if (0<l<=length(ecp.soblocks))
        res .+= s*lblock_on_grid(cpars, grid, ecp.soblocks[l])#*2/(l*(l+1))
    end
    res
end

struct GRECPadd
   ks
   funcs
   lblocks
end

function (grecp::GRECPadd)(cpars, grid, kappa)
    j = abs(kappa)-1/2
    l = Int(j + sign(kappa)/2)
    res = zeros(eltype(grid), length(grid.xs), length(grid.xs))
    ind = findfirst(grecp.ks .== kappa)
end    
end
