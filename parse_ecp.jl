""" provides structs and functions for parsing ecp in dirac format"""
module ParseECP
export ECPL, from_dirac
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
"""function to parse header of ecp in dirac format
header is string: 
ECP N lmax nso

- ECP is the literal
- N -- float value of number of excluded electrons
- lmax - maximal L value
- nso - number of spin-orbit blocks
"""
function parse_header(str)
    aux = split(str)
    @assert aux[1] == "ECP"
    N = parse(Float64, aux[2])
    lmax = parse(Int, aux[3])
    nso = parse(Int, aux[4])
    (N=N, lmax=lmax, nso=nso)
end
get_expsnum(line) = parse(Int64, line)

function get_exp(line) 
    aux = split(line)
    pow = parse(Int, aux[1])
    α = parse(Float64, aux[2])
    c = parse(Float64, aux[3])
    (α, pow, c)
end


function nonempty_without_comments(io)
    res = Channel() do ch
        while (!eof(io))
            line = readline(io)
            line = strip(split(line, "#")[1])
            if (length(line) == 0)
                continue
            end
            put!(ch, line)
        end
    end
    res 
end

"""get one l block from io stream"""
function get_lblock(stream)
    line = ""
    nexps = get_expsnum(take!(stream))
    as = zeros(nexps)
    ps = zeros(Int64, nexps)
    cs = zeros(nexps)
    for kk=1:nexps
        as[kk], ps[kk], cs[kk] = get_exp(take!(stream))
    end
    (αs = as, pows=ps, cs=cs)
end

"""parses ECPs entry in Dirac format"""
function from_dirac(T, io)
    stream = nonempty_without_comments(io)
    header = parse_header(take!(stream))
    lblocks = ECPL{T}[]
    as, ps, cs = get_lblock(stream) # ul part
    push!(lblocks, ECPL(as, ps, cs))
    for l=0:header.lmax-2 #semilocal parts
        as, ps, cs = get_lblock(stream) 
        push!(lblocks, ECPL(as, ps, cs))
    end
    soblocks = ECPL{T}[]
    for kk=1:header.nso
        as, ps, cs = get_lblock(stream) 
        push!(soblocks, ECPL(as, ps, cs))
    end
    (N=header.N, lblocks, soblocks)
end
end
