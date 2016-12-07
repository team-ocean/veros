import numpy

def u_centered_grid(dyt, n):
    yu = numpy.zeros(n)
    yt = numpy.zeros(n)
    dyu = numpy.zeros(n)
    for i in xrange(1, n):
        yu[i] = yu[i-1] + dyt[i]
        yt[i] = 2*yu[i-1] + yt[i-1]
    dyu[:n-1] = yt[1:] - yt[:n-1]
    dyu[n-1] = 2*dyt[n] - dyu[n-1]
    return (yu, yt, dyu)

def calc_grid():
    pass

def calc_beta():
    """
    --------------------------------------------------------------
     calculate beta = df/dy
    --------------------------------------------------------------
    """
    for j in xrange(js_pe,je_pe+1): # j=js_pe,je_pe
        beta[:,j] = 0.5*(  (coriolis_t[:,j+1]-coriolis_t[:,j])/dyu[j] + (coriolis_t[:,j]-coriolis_t[:,j-1])/dyu[j-1] )

def calc_topo():
    """
    --------------------------------------------------------------
     calulate masks, total depth etc
    --------------------------------------------------------------
    """

    """
    --------------------------------------------------------------
     close domain
    --------------------------------------------------------------
    """
    if my_blk_j == 1:
        kbot[:,1-onx:0] = 0
    if my_blk_j == n_pes_j:
        kbot[:,ny+1:ny+onx] = 0  
    if not enable_cyclic_x:
        if my_blk_i == 1:
            kbot[1-onx:1,:] = 0
        if my_blk_i == n_pes_i:
            kbot[nx+1:nx+onx+1,:] = 0

    """
    --------------------------------------------------------------
     Land masks
    --------------------------------------------------------------
    """
    maskT = 0.0
    for k in xrange(1, nz+1): # k=1,nz
        for j in xrange(js_pe-onx, je_pe+onx): # j=js_pe-onx,je_pe+onx
            for i in xrange(is_pe-onx, ie_pe+onx): # i=is_pe-onx,ie_pe+onx
                if kbot[i,j] != 0 and kbot[i,j] <= k:
                    maskT[i,j,k] = 1.0
    maskU = maskT
    for i in xrange(is_pe-onx, ie_pe+onx): # i=is_pe-onx,ie_pe+onx-1
        maskU[i,:,:] = min(maskT[i,:,:], maskT[i+1,:,:])
    maskV = maskT
    for j in xrange(js_pe-onx, je_pe+onx): # j=js_pe-onx,je_pe+onx-1
        maskV[:,j,:] = min(maskT[:,j,:], maskT[:,j+1,:])
    maskZ = maskT
    for j in xrange(js_pe-onx, je_pe+onx): # j=js_pe-onx,je_pe+onx-1
        for i in xrange(is_pe-onx, ie_pe+onx): # i=is_pe-onx,ie_pe+onx-1
            maskZ[i,j,:] = min(maskT[i,j,:],maskT[i,j+1,:],maskT[i+1,j,:])
    maskW = maskT
    for k in xrange(1, nz): # k=1,nz-1
        maskW[:,:,k] = min(maskT[:,:,k],maskT[:,:,k+1])
    """
    --------------------------------------------------------------
     total depth
    --------------------------------------------------------------
    """
    ht=0.0
    hu=0.0
    hv=0.0
    for k in xrange(1, nz+1):
        ht = ht+maskT[:,:,k]*dzt[k]
        hu = hu+maskU[:,:,k]*dzt[k]
        hv = hv+maskV[:,:,k]*dzt[k]
    hur[hu != 0.0] = 1./hu[hu != 0.0]
    hvr[hv != 0.0] = 1./hv[hv != 0.0]

#TODO: you are here

def calc_initial_conditions():
    pass

def ugrid_to_tgrid():
    pass

def vgrid_to_tgrid():
    pass

def solve_tridiag(a, b, c, d, n):
    cp = numpy.zeros(n)
    dp = numpy.zeros(n)

    # initialize c-prime and d-prime
    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]

    # solve for vectors c-prime and d-prime
    for i in xrange(1, n):
        m = b[i] - cp[i-1] * a[i]
        fxa = 1.0 / m
        cp[i] = c[i] * fxz
        dp[i] = d[i]-dp[i-1]*a[i]
    x[n-1] = dp[n-1]
    for i in xrange(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]

    return x

def calc_diss():
    pass
