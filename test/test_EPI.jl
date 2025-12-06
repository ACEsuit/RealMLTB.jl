using Test
using FastGaussQuadrature
using LinearAlgebra
using Molecules
using GaussianBasis
import GaussianBasis: num_basis

function def_EPI_1e3c(BS::BasisSet)
   out = zeros(BS.nbas, BS.nbas, BS.nbas)

   Nvals = num_basis.(BS.basis)
   Nmax = maximum(Nvals)

   # Offset list for each shell, used to map shell index to AO index
   ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:BS.nshells]

   ijk = [[i, j, k] for i = 1:BS.nshells, j = 1:BS.nshells, k = 1:BS.nshells]

   buf = zeros(Cdouble, Nmax^3)
   for (i, j, k) in ijk
      Ni = Nvals[i]
      Nj = Nvals[j]
      Nk = Nvals[k]
      Nijk = Ni * Nj * Nk
      ioff = ao_offset[i]
      joff = ao_offset[j]
      koff = ao_offset[k]

      I = (ioff+1):(ioff+Ni)
      J = (joff+1):(joff+Nj)
      K = (koff+1):(koff+Nk)

      generate_V_triple!(buf, BS, i, j, k)

      vals = reshape(view(buf, 1:Nijk), Ni, Nj, Nk)
      out[I, J, K] = vals
   end
   return out
end

function def_V_triple!(out, B1::CartesianShell, B2::CartesianShell, B3::CartesianShell)
   out .= 0.0

   am1 = B1.l
   am2 = B2.l
   am3 = B3.l

   A = B1.atom.xyz .* ang2bohr
   B = B2.atom.xyz .* ang2bohr
   C = B3.atom.xyz .* ang2bohr

   V(i,j,k,a,b,c,dim) = V3(i,j,k,a,b,c,A[dim],B[dim],C[dim])

   N1 = num_basis(B1)
   N2 = num_basis(B2)

   for (ca, a) in zip(B1.coef, B1.exp)
      for (cb, b) in zip(B2.coef, B2.exp)
         for (cc, c) in zip(B3.coef, B3.exp)
            prefac = ca * cb * cc
            index1 = 1
            for l1 in am1 : -1 : 0, n1 in 0 : am1 - l1
               m1 = am1 - l1 - n1
               index2 = 1
               for l2 in am2 : -1 : 0, n2 in 0 : am2 - l2
                  m2 = am2 - l2 - n2
                  index3 = 1
                  for l3 in am3 : -1 : 0, n3 in 0 : am3 - l3                     
                     m3 = am3 - l3 - n3
                     index = index1 + N1*(index2-1) + N1*N2*(index3-1)
                     out[index] += V(l1,l2,l3,a,b,c,1) * V(m1,m2,m3,a,b,c,2) * V(n1,n2,n3,a,b,c,3) * prefac
                     index3 += 1
                  end
                  index2 += 1
               end
               index1 += 1
            end
         end
      end    
   end
   return out
end

function def_V_triple!(out, BS::BasisSet, s1, s2, s3)
   def_V_triple!(out, BS.basis[s1], BS.basis[s2], BS.basis[s3])
end

function V3(i::Int, j::Int, k::Int, a, b, c, A, B, C)
   p = a + b
   q = p + c
   P = (a * A + b * B) / p
   Q = (p * P + c * C) / q

   µ = a * b / p
   ν = p * c / q
   K = exp(-µ * (A-B)^2 - ν * (P-C)^2)

   x, w = gausshermite(5)

   f(x) = K * (x - A)^i * (x - B)^j * (x - C)^k
   fst(x) = f(x/sqrt(q) + Q)/sqrt(q)
   
   dot(w, fst.(x))
end

atoms = Molecules.parse_string("""
C   -2.131551124300    2.286168823700    0.000000000000
H   -1.061551124300    2.286168823700    0.000000000000
H   -2.488213906200    1.408104616400    0.496683911300
H   -2.488218762100    2.295059432700   -1.008766153900
H   -2.488220057000    3.155340844300    0.512081313000""")

bs1 = BasisSet("6-31g", atoms, spherical=false, lib=:acsint)
bs2 = BasisSet("sto-3g", atoms, spherical=false, lib=:acsint)

@testset "Triple Shell Integrals" begin
   for BS in [bs1, bs2]
      Nmax = maximum(num_basis.(BS.basis))
      out = zeros(Nmax^3)
      ref = similar(out)
      ijk = find_unique_ijk(BS.nshells)
      for (i,j,k) in ijk
         generate_V_triple!(out, BS, i, j, k)
         def_V_triple!(ref, BS, i, j, k)
         @test isapprox(out, ref, atol=1e-12)
      end
   end
end

@testset "Effective Potential Integrals" begin
   for BS in [bs1, bs2]
      out = EPI_1e3c(BS)
      ref = def_EPI_1e3c(BS)
      @test isapprox(out, ref)
   end
end
