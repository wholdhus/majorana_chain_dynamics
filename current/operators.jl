using ITensors, ITensorMPS

function make_H0(L, sites, bc)
    osI = OpSum()
    for j in 1:L
        osI -= "Z", j
    end    
    for j in 1:L-1
        osI -= "X", j, "X", j+1
    end
    
    if bc == "PBC"
        osI -= "X", L, "X", 1
    elseif bc == "APBC"
        osI += "X", L, "X", 1
    end
    return MPO(osI, sites)
end

function make_H3(L, sites, bc)
    os3 = OpSum()
    for j in 1:L-2
        os3 += "Z", j, "X", j+1, "X", j+2
        os3 += "X", j, "X", j+1, "Z", j+2
    end
    if bc == "PBC"
        os3 += "Z", L-1, "X", L, "X", 1
        os3 += "X", L-1, "X", L, "Z", 1

        os3 += "Z", L, "X", 1, "X", 2
        os3 += "X", L, "X", 1, "Z", 2
    elseif bc == "APBC"
        os3 -= "Z", L-1, "X", L, "X", 1
        os3 += "X", L-1, "X", L, "Z", 1

        os3 += "Z", L, "X", 1, "X", 2
        os3 -= "X", L, "X", 1, "Z", 2
    end
    return MPO(os3, sites)
end

function local_H(t1, g, s1, s2, s3; signs=[1.0, 1.0, 1.0, 1.0])
    X_part = -signs[1]*op("Z", s1) * op("Id", s2) * op("Id", s3)
    H = t1*X_part
    if signs[2] != 0.0
        ZZ_part = -signs[2]*op("X", s1) * op("X", s2) * op("Id", s3)
        H += t1*ZZ_part
    end
    # Build term for Three-spin part H3
    if signs[3] != 0
        H += g*signs[3]*op("Z", s1) * op("X", s2) * op("X", s3) 
    end
    if signs[4] != 0
        H += g*signs[4]*op("X", s1) * op("X", s2) * op("Z", s3)
    end
    return H
end
