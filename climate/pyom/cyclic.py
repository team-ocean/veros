def setcyclic_xy(p1, enable_cyclic_x, nx):
    """
    --------------------------------------------------------------
           set cyclic boundary conditions for 2D array
    --------------------------------------------------------------
    """
    #use main_module
    #implicit none
    #integer, intent(in) :: is_,ie_,js_,je_
    #real*8, intent(inout) :: p1(is_:ie_,js_:je_)
    #integer :: j,i

    if enable_cyclic_x:
        for i in xrange(2): #i=1,onx
            p1[nx+2+i,:] = p1[i+2,:]
            p1[1-i,:]  = p1[nx-i+1,:]

def setcyclic_xyz(a, enable_cyclic, nx, nz):
    """
    --------------------------------------------------------------
           set cyclic boundary conditions for 3D array
    --------------------------------------------------------------
    """
    #use main_module
    #implicit none
    #integer:: is_,ie_,js_,je_,nz_
    #real*8 :: a(is_:ie_,js_:je_,nz_)
    #integer :: k
    for k in xrange(nz): #k=1,nz
        setcyclic_xy(a[:,:,k], enable_cyclic, nx)

def setcyclic_xyp(np, p1, enable_cyclic_x, nx):
    """
    --------------------------------------------------------------
           set cyclic boundary conditions for 3D array
    --------------------------------------------------------------
    """
    #use main_module
    #implicit none
    #integer:: is_,ie_,js_,je_,np
    #real*8 :: p1(is_:ie_,js_:je_,np)
    #integer :: i

    p1[:,:,1 ] = p1[:,:,np-1]
    p1[:,:,np] = p1[:,:,2]
    if enable_cyclic_x:
        for i in xrange(onx): #i=1,onx
            p1[nx+i,:,:] = p1[i  ,:,:]
            p1[1-i,:,:]  = p1[nx-i+1,:,:]
