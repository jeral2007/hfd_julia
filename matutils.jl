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
end   
