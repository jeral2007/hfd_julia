module GausOpt
export get_coefs, make_proj_vecs, make_target, scale, one_exp_vec
using LinearAlgebra
using Optim
function gaus_smat(cpars, grid, alphas, l)
    I1(α) = dot(grid.ws, grid.xs.^(2l+2).*exp.(-α .* grid.xs.^2))
    smat = broadcast((α, β)-> I1(α+β)/sqrt(I1(2α)*I1(2β)),
        alphas', alphas)
    norms = sqrt.(I1.(2e0*alphas))
    (smat=qr(smat, ColumnNorm()), norms=norms)
end

function put_rhs!(rhs, cpars, grid, alphas, l, vec)
    rhs = zeros(eltype(vec), size(alphas))
    for ii in eachindex(rhs)
        rhs[ii] = dot(grid.ws, grid.xs.^(l+1).*vec.*exp.(-alphas[ii] .* grid.xs.^2))
    end
    rhs
end

function get_coefs(cpars, grid, alphas, l, vecs)
    smat, norms = gaus_smat(cpars, grid, alphas, l)
    rhs = zeros(eltype(vecs), length(alphas), size(vecs, 2))
    for ii=1:size(vecs, 2)
        #@show put_rhs!(rhs[:, ii], cpars, grid, alphas, l, vecs[:, ii])
        @views rhs[:, ii] .= put_rhs!(rhs[:, ii], cpars, grid, alphas, l, vecs[:, ii])
        @views rhs[:, ii] ./= norms
    end
   # @show smat
   # @show rhs
    coefs = smat\rhs
    (coefs=coefs, norms=norms)
end

function make_proj_vecs(cpars, grid, cfn, alphas, l)
    proj_vecs = zeros(eltype(grid.xs), cpars.N, size(cfn.coefs, 2))
    for ii=1:cpars.N, jj = 1:size(cfn.coefs, 2)
        x = grid.xs[ii]
        for kk=1:size(cfn.coefs, 1)
            c = cfn.coefs[kk, jj]/cfn.norms[kk]
            proj_vecs[ii, jj] += c*exp(-alphas[kk]*x^2) * x^l
        end
    end
    proj_vecs
end

scale_all(alphas, ts) = 1/sqrt(2) .* alphas.*(1  .+ ts.^2 ./ (1e0.+ts.^2))
function scale(alphas, ts; active=nothing)
    if active==nothing
        alphas_new = scale_all(alphas, ts)
    else
        alphas_new = copy(alphas)
        alphas_new[active] .= scale_all(alphas[active], ts)
    end
    alphas_new
end   

function make_target(cpars, grid, alphas, l, vecs; active= nothing)
    nrms_vecs = sqrt.(grid.ws' * vecs.^2)
    function target(ts)
        alphas_new = scale(alphas, ts; active)
        cfn = get_coefs(cpars, grid, alphas_new, l, vecs)
        pvecs = make_proj_vecs(cpars, grid, cfn, alphas_new, l)
        nrms = sqrt.(grid.ws' *(pvecs.^2 .* grid.xs.^2))
        @views pvecs .= pvecs.*grid.xs .* vecs ./ nrms/nrms_vecs
       # nrm = 
        ress = grid.ws' * pvecs
        #@show ress
        -sum(ress) 
    end
end

function one_exp_vec!(cpars, grid, vec, l)
    nrm = dot(grid.ws, vec.^2)
    @show nrm
    if nrm.<eps(vec[1])
        return (alpha= 0e0, err=-1)
    end
    rho2 = dot(grid.ws, grid.xs.^2 .* vec.^2)/nrm
    @show rho2
    #@views vec ./= sqrt(nrm)
    Ia(α) = dot(grid.ws, grid.xs.^(2l+2) .*exp.(-α .* grid.xs.^2))
    Iav(α) = dot(grid.ws, grid.xs.^(l+1) .*exp.(-α .* grid.xs.^2) .* vec)
    function tgt(a)
        an = 3/rho2/(2a[1]^2+1)
        Iav(an)/Ia(2an)
    end
    sol = optimize(tgt, [1e0])
    α = 3/rho2/(2Optim.minimizer(sol)[1]^2 + 1)
    @views vec .-= grid.xs.^(l+1) .* exp.(-α .* grid.xs.^2) .* Iav(α)/Ia(2α)
    α[1], 0
end
end