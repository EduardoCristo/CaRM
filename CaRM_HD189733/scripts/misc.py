def boltoint(inputlist):
    b = []
    nb = []
    for x in range(len(inputlist)):
        b += [[]]
        nb += [[]]
        for y in inputlist[x]:
            if y == True:
                b[x] += [1]
                nb[x] += [0]
            else:
                b[x] += [0]
                nb[x] += [1]
    return(b, nb)


def multiply(a, b):
    if len(a) != len(b):
        import sys
        sys.exit(
            "Dimensions in the multiplication are not the same! Check the input parameters.")
    c = []
    for x in range(len(a)):
        print(x)
        c += [[]]
        for y in range(len(a[x])):
            c[x] += [a[x][y]*b[x][y]]
    return(c)


#parpos=boltoint([[True, True, True, False, True, True, False, False, False, False, True, True], [True, False, False, False, False, False, False, False, False, False, True, False]])
"""
pguess=[
[-1.47400000e+01,1.19497102e-01 , 8.51000000e-02 , 8.76000000e+00,
 8.65900000e+01,-5.00000000e+00,6.50478516e-01,4.70000000e+00,
 9.00000000e+01,5.82144273e-06,0.00000000e+00 -1.60000000e+01],
[-1.47400000e+01,1.19497102e-01,8.51000000e-02,8.76000000e+00,
8.65900000e+01 -5.00000000e+00,6.50478516e-01,4.70000000e+00,
9.00000000e+01,5.82144273e-06,0.00000000e+00 ,-1.60000000e+01]]

print(multiply(pguess,parpos[1]))
"""
