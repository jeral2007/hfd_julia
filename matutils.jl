module MatUtils
using LinearAlgebra
export refine!, accelerate!
function refine_iter!(A, B, y, u, r, mat;dump)
    u .= B * y
    r .= A * y
    λ = dot(y, r)/dot(y, u)
    @views begin 
        r .-= λ .* u
        mat .= A .- λ .* B
        y .-= qr!(mat, ColumnNorm())\(dump .* r)
        y ./= norm(y)
    end
    (λ, norm(r))
end
"""refines eigenvector y iteratively by perturbation theory"""
function refine!(A, B, y;maxiter=5, tol=1e-7, dump = 0.5)
    u = similar(y)
    r = similar(y)
    mat = similar(A)
    errs = eltype(y)[]
    λs = eltype(y)[]
    for iter=1:maxiter
        λ, err = refine_iter!(A, B, y, u, r, mat; dump)
        push!(λs, λ)
        push!(errs, err)
        if (err<tol) 
            break
        end
    end
    λs, errs
end
""" aitken δ2 accelerator for matrices ( element by element)
Arguments:
- val -- array with present values
- oldval -- array with previous values
- old2val -- array with previous previous values
- eps - if absolute value of second difference < eps - do not accelerate
"""
function accelerate!(val, oldval, old2val; eps=1e-10, eps_corr = 1e-2)
   for ai in eachindex(val)
       d1 = oldval[ai] - old2val[ai]
       d2 = old2val[ai] + val[ai] - 2*oldval[ai]
       if (-eps < d2 < eps)
           continue
       end
       corr = d1^2/d2
       if abs(corr)>eps_corr
          continue
       end
       val[ai] = old2val[ai] - corr
   end
end

"""kmeans clusterization of arr. The kl_inds array is the initial clusterization of arr. 
Axes of kl_inds must be same to that of arr. kl_inds[kk] is the cluster number of kk-th element of arr."""
function kmeans!(kl_inds, arr::Vector{T}) where {T<:Real}
    kl_labs = sort(unique(kl_inds))
    means = zeros(T, size(kl_labs, 1))
    d2 = similar(means)
    changed = true
    while changed 
        changed = false
        for kk=1:size(kl_labs, 1)
            mask = kl_inds .== kl_labs[kk]
            means[kk] = sum(arr[mask])/count(mask)
            d2[kk] = sum(arr[mask].^2)/count(mask)
        end
        @views d2 .-= means .^2
        for ii in eachindex(arr)
            dists = abs.(arr[ii] .- means)
            ki = argmin(dists)
            #@show ii ki kl_labs[ki] kl_inds[ii]
            changed |= kl_labs[ki] != kl_inds[ii]
            kl_inds[ii] = kl_labs[ki]
        end
    end
    return sum(d2)
end
end   
