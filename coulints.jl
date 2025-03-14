"""evaluates radial integral of dens * r<^k/r^k+1. modifies density
result is multiplied by r"""
function rad_int!(cpars, grid, dens, k, res)
    @views dens .*=grid.xs^k
    
