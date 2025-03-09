module PolyNodes
using LinearAlgebra
struct Lagrange{T} 
    xs :: Vector{T}
    ys :: Vector{T}
end
function (lagr::Lagrange{T})(x::T) where {T}
    res::T = 0e0
    for ii=1:length(lagr.xs)
        prod=lagr.ys[ii]
        for jj=1:length(lagr.xs)
            if ii==jj
                continue
            end
            prod *= (x-lagr.xs[jj])/(lagr.xs[ii]-lagr.xs[jj])
        end
        res += prod
    end
    return res
end

function diffOp(xs)
    N = length(xs)
    res = zeros(eltype(xs), N, N)
    oton = Vector(1:N)
    for kk=1:N, ii=1:N
        mask = (oton .!= ii) .& (oton .!= kk)
        if kk!=ii
            res[kk, ii] = prod((xs[kk].-xs[mask])./(xs[ii].-xs[mask]))
            res[kk, ii] /=(xs[ii] - xs[kk])
        else
            res[ii, ii] = sum(1e0./(xs[ii] .- xs[mask]))
        end
    end
    res
end
alphas_leg(N) = zeros(N)
gammas_leg(N) = [l^2/(2l+1e0)/(2l-1e0) for l=1:N-1]

function nodes(alphas, gammas, wtot)
    mat = diagm(0=>alphas, 1=>sqrt.(gammas), -1=>sqrt.(gammas))
    ts, aux = eigen(mat)
    ws = wtot.*aux[1, :] .^ 2
    (ts=ts, ws=ws)
end

function caley_trans(tws, k=1e0)
    ts, ws = tws
    xs = k*(1e0 .+ ts) ./ (1e0 .- ts)
    wxs = 2e0.* ws .* k ./ (1 .-ts).^2
    (ts=xs, ws=wxs)
end
end
