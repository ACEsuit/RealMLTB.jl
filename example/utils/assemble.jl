using LinearAlgebra, StaticArrays

"""
assemble_Wv(val)

General-purpose assembly function. Automatically handles arbitrary angular momenta 
(S, P, D, F, G...) based on the structure of `val`.

Assumptions:
  - val[1] corresponds to L=0 (S), elements are Float64 (or length-1 vectors).
  - val[2] corresponds to L=1 (P), elements are SVector{3}.
  - val[3] corresponds to L=2 (D), elements are SVector{5}.
  - val[k] corresponds to L=k-1, element length is 2(k-1)+1.
"""
function assemble_Wv(val_tuple)
    n_atoms = size(val_tuple[1], 1)


    total_len = 0
    for (L_idx, block) in enumerate(val_tuple)
        L = L_idx - 1
        n_funcs = 2 * L + 1
        n_shells = size(block, 2)
        total_len += n_atoms * n_shells * n_funcs
    end

    Wv = zeros(Float64, total_len)
    cursor = 1

    perm_L1 = SVector{3, Int}(3, 1, 2)
    perm_L2 = SVector{5, Int}(1, 2, 3, 4, 5) 

    @inbounds for i in 1:n_atoms
        for (L_idx, block) in enumerate(val_tuple)
            L = L_idx - 1
            n_shells = size(block, 2)
            
            for j in 1:n_shells
                v = block[i, j]

                if L == 0 
                    Wv[cursor] = v
                    cursor += 1
                    
                elseif L == 1
                    Wv[cursor]     = v[perm_L1[1]]
                    Wv[cursor + 1] = v[perm_L1[2]]
                    Wv[cursor + 2] = v[perm_L1[3]]
                    cursor += 3
                    
                elseif L == 2
                    for k in 1:5
                         Wv[cursor + k - 1] = v[perm_L2[k]]
                    end
                    cursor += 5
                    
                else
                    for k in eachindex(v)
                        Wv[cursor + k - 1] = v[k]
                    end
                    cursor += length(v)
                end
            end
        end
    end

    return Wv
end