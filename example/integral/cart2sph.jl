using SpecialFunctions

"""
	cart2sph_pure(lx::Int, ly::Int, lz::Int, m::Int)

The coefficient that transforms normalized Cartesian GTOs
to normalized pure spherical harmonic GTOs.
	
    gY(l,m) = \\sum_{lx+ly+lz=l} gO(lx,ly,lz) * c(l,m,lx,ly,lz)
	 
gY(l,m) is a pure spherical harmonic GTO and gO(lx,ly,lz) is a Cartesian GTO
with Condon-Shortley phase convention.
	 
Ref: H. B. Schlegel and M. J. Frisch, Int. J. Quant. Chem., 54(1955), 83-87. Eq. (15)
"""
function cart2sph_pure(lx::Int, ly::Int, lz::Int, m::Int)
	if isodd(lx+ly+m)
		return 0
	end

	abs_m = abs(m)
	l = lx + ly + lz
	j = div(lx + ly - abs_m, 2)
    num = factorial(2lx) * factorial(2ly) * factorial(2lz) * 
		  factorial(l) * factorial(l-abs_m)
	den = factorial(2l) * factorial(lx) * factorial(ly) * factorial(lz) * factorial(l+abs_m)
    c0 = sqrt(num/den) * (0.5^l) / factorial(l)

	cpi = 0
    for i in max(0,j):div(l-abs_m,2)
        cp0 = binomial(l, i) * binomial(i, j) * x_factorial(1, 2l - 2i, l - abs_m - 2i)
		cpi += alt_sign(i) * cp0
	end            
    c0 *= cpi

	cpk = 0
	for k in max(0,div(lx-abs_m+1,2)):min(j,div(lx,2))
		cp0 = binomial(j, k) * binomial(abs_m, lx-2k)
		cpk += alt_sign((abs_m-lx+2k)//2) * cp0
	end

	# (-)^m introduced by Condon-Shortley phase convention in Legendre function.
	# IJQC_54_83 does not have this phase parameter
    if m >= 0
		c = alt_sign(m) * c0 * cpk
    else
	# m<0, Y(l,-m) = (-)^m Y(l,m)^*
        c = c0 * conj(cpk)
	end

    return c
end

"""
    xyz2sph_pure(lx, ly, lz, m)

Factor of xyz component for normalized pure spherical harmonic functions

     r^l*Y(l,m) = \\sum_{lx+ly+lz=l} x^lx*y^lyz^lz * c(l,m,lx,ly,lz)

Y(l,m) is a pure spherical harmonic function with Condon-Shortley phase
convention

Ref: H. B. Schlegel and M. J. Frisch, Int. J. Quant. Chem., 54(1995), 83-87.  Eq. (7)
"""
function xyz2sph_pure(lx::Int, ly::Int, lz::Int, m::Int)
    if isodd(lx + ly + m)
        return 0
    end

    abs_m = abs(m)
    l = lx + ly + lz
    j = div(lx + ly - abs_m, 2)
    c0 = sqrt(x_factorial((2l + 1) / (4pi), l - abs_m, l + abs_m)) * (0.5^l) / factorial(l)

	cpi = 0
    for i in max(0,j):div(l-abs_m,2)
        cp0 = binomial(l, i) * binomial(i, j) * x_factorial(1, 2l - 2i, l - abs_m - 2i)
		cpi += alt_sign(i) * cp0
	end
	c0 = c0 * cpi

	cpk = 0
	for k in max(0,div(lx-abs_m+1,2)):min(j,div(lx,2))
        cp0 = binomial(j,k) * binomial(abs_m, lx-2k)
		cpk += alt_sign((abs_m-lx+2k)//2) * cp0
	end

	# Condon-Shortley phase convention
	if m >= 0
		c = alt_sign(m) * c0 * cpk
    else
        c = c0 * conj(cpk)
	end

	return c
end

"""
    xyz2sph_real(lx::Int, ly::Int, lz::Int, m::Int)

Factor of xyz component for normalized real spherical harmonic functions.

     r^l*Y(l,m) = \\sum_{lx+ly+lz=l} x^lx y^ly z^lz c(l,m,lx,ly,lz)

Y(l,m) is a real spherical harmonic function with Condon-Shortley phase convention.

Ref: H. B. Schlegel and M. J. Frisch, Int. J. Quant. Chem., 54(1995), 83-87.
"""
function xyz2sph_real(lx::Int, ly::Int, lz::Int, m::Int)
    if isodd(lx + ly + m)
        return 0
    end

    abs_m = abs(m)
    l = lx + ly + lz
    j = div(lx + ly - abs_m, 2)
    c0 = sqrt(x_factorial((2l + 1) / (4pi), l - abs_m, l + abs_m)) * (0.5^l) / factorial(l)

    cpi = 0
    for i in max(0, j):div(l - abs_m, 2)
        cp0 = binomial(l, i) * binomial(i, j) * x_factorial(1, 2l - 2i, l - abs_m - 2i)
        cpi += alt_sign(i) * cp0
    end
    c0 = c0 * cpi

    cpk = 0
    for k in max(0, div(lx - abs_m + 1, 2)):min(j, div(lx, 2))
        cp0 = binomial(j, k) * binomial(abs_m, lx - 2k)
        r = (abs_m - lx + 2k) % 4

		if m >= 0
        	if r == 0 
            	cpk += cp0
			elseif r == 2
            	cpk -= cp0
			end
        else
			if r == 1
            	cpk -= cp0
			elseif r == 3
            	cpk += cp0
			end
		end
    end

	if m == 0
		c = c0 * cpk
	else
    	c = sign(m) * sqrt(2) * c0 * cpk
	end

    return c
end

"""
	cart2sph(l::Int)

Return 2D array that transforms normalized cartesian basis 
to normalized real spherical basis
"""
function cart2sph(l)
    if l == 0
        return Matrix{Float64}(I, 1, 1)
	elseif l == 1
        return Matrix{Float64}(I, 3, 3)
    else
        u1 = Float64[]
        for lx in l:-1:0
            for ly in (l-lx):-1:0
                lz = l - lx - ly
				for m in -l:l
					push!(u1, xyz2sph_real(lx, ly, lz, m))
				end
        	end
    	end
    	ncart = div((l + 1) * (l + 2), 2)
		nsph = 2l+1
        u1 = transpose(reshape(u1, nsph, ncart))
    	return u1
	end
end

function cart2sph_coeff(bs::BasisSet)
    @assert bs.lib isa GaussianBasis.ACSint

    ls = basis_angular.(bs.basis)
    max_l = maximum(ls)
    c2s_l = [cart2sph(l) for l in 0:max_l]

	c2s = Matrix[]
	for l in ls
		push!(c2s, c2s_l[l+1])
	end

	return block_diag(c2s)
end

basis_angular(B::BasisFunction) = B.l

# compute (-1)^m, where m≥0 is an integer
function alt_sign(m::Int)
	isodd(m) ? -1 : 1
end

# compute (-1)^m, where m≥0 is a half integer
function alt_sign(m::Rational{Int})
	if m.den == 2
		if m.num % 4 == 1
			return im
		else
			return -im
		end
	elseif m.den == 1
		return alt_sign(m.num)
	end
end

# Accurately compute `x * n! / m!` for large n
@inline x_factorial(x, n, m) = exp(log(x) - logfactorial(m) + logfactorial(n))

function block_diag(matrices::Vector{<:AbstractMatrix})
    rows = sum(size.(matrices, 1))
    cols = sum(size.(matrices, 2))
    
    B = zeros(eltype(matrices[1]), rows, cols)
    
    current_row = 1
    current_col = 1
    
    for M in matrices
        R, C = size(M)
        B[current_row:(current_row + R - 1), current_col:(current_col + C - 1)] = M
        
        current_row += R
        current_col += C
    end
    return B
end
