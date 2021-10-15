
#
# solving the linear stability problem
# for U(z), V(z), and B(z)
#
# version for primitive equation is pe
# version for quasi-geostrophic approx. is qg
#
# constant grid spacing is assumed for both

def pe(U,V,B0,dz,kx,ky,betax,betay,fh,f0,hx,hy,f_filter=0.5):
  # solution for primitive equations
  import numpy as np
  import sys
  cs = 150 # speed of sound, artificially reduced
  N=U.shape[0]
  # derivatives of U,V and B
  UZ=np.zeros(N,'d');VZ=np.zeros(N,'d');BZ=np.zeros(N,'d')
  for n in range(1,N-1):
   UZ[n]=(U[n+1]-U[n-1])/(2*dz)
   VZ[n]=(V[n+1]-V[n-1])/(2*dz)
   BZ[n]=(B0[n+1]-B0[n-1])/(2*dz)
  UZ[0]=UZ[1];UZ[-1]=UZ[-2];
  VZ[0]=VZ[1];VZ[-1]=VZ[-2]; 
  BZ[0]=BZ[1];BZ[-1]=BZ[-2];
  # allocate some variables
  I=complex(0,1)
  A  = np.zeros((5,5,N),'Complex64')
  B  = np.zeros((5,5,N),'Complex64')
  C  = np.zeros((5,5,N),'Complex64')
  AA = np.zeros((5*N,5*N),'Complex64')
  om_max= np.zeros((kx.shape[0],ky.shape[0]),'Complex64');
  omax  = complex(0,0)
  AAmax = np.zeros((5*N,5*N),'Complex64')
  kmax  = 0; lmax  = 0
  # enter main loop
  for j in range(ky.shape[0]): # loop over meridional wavelength
    sys.stdout.write('\b'*21+'calculating j=%3i/%3i'%(j,ky.shape[0]) )
    sys.stdout.flush()
    for i in range(kx.shape[0]): # loop over zonal wavelength

      Uk   = kx[i]*U+ky[j]*V   # k \cdot \v U
      UZk  = kx[i]*UZ+ky[j]*VZ # k \cdot \v U'
      UZhk =-kx[i]*VZ+ky[j]*UZ # k \cdot \rvec{\v U'}

      kh2 = ky[j]**2 + kx[i]**2 + 1e-18
      fhk = ( -betay*kx[i]+betax*ky[j] )/kh2
      fk  = (  betax*kx[i]+betay*ky[j] )/kh2

      for n in range(N): # loop over depth
        np1 = min(N-1,n+1)
        nm  = max(0,n-1)

        B[0,2,n]=  (ky[j]*fh+UZhk[n])/(2*kh2)
        B[1,2,n]= -(kx[i]*fh+UZk[n])/(2*kh2)
        B[3,2,n]= I*BZ[n]/2
        B[4,2,n]= -cs**2*I/dz

        C[2,0,n]=  ky[j]*fh/2
        C[2,1,n]= -kx[i]*fh/2
        C[2,3,n]= -I/2
        C[2,4,n]=  I/dz

        A[0,:,n]= [ Uk[n]+fhk,I*f0,(ky[j]*fh+UZhk[n])/(2*kh2),0,0 ]
        A[1,:,n]= [-I*f0 ,Uk[n], -(kx[i]*fh+UZk[n])/(2*kh2),0,I]
        A[2,:,n]= [ ky[j]*fh/2 ,-kx[i]*fh/2 ,(Uk[n]+Uk[np1])/2,-I/2,-I/dz ]
        A[3,:,n]= [ -f0*UZk[n],-f0*UZhk[n],I*BZ[n]/2,Uk[n], 0  ]
        A[4,:,n]= [ 0, -I*cs**2*kh2 ,  I*cs**2/dz, 0   , 0 ]

      # upper boundary condition
      A[2,:,-1]=0; B[2,:,-1]=0; C[2,:,-1]=0;
      A[:,2,-1]=0
      C[:,2,-1]=0
      C[:,2,-2]=0

      # lower boundary condition
      #B[:,3,1]=0; 
      #A[:,0,0]=A[:,0,0]-I*(-kx[i]*hy+ky[j]*hx)*B[:,2,0]
      #A[:,1,0]=A[:,1,0]+I*( kx[i]*hx+ky[j]*hy)*B[:,2,0]

      A[:,0,0]=A[:,0,0]-2*I*(-kx[i]*hy+ky[j]*hx)*B[:,2,0]
      #A[:,1,0]=A[:,1,0]+2*I*( kx[i]*hx+ky[j]*hy)*B[:,2,0]
      A[:,2,0]=A[:,2,0]-B[:,2,0]

      # build large matrix
      n1 = 0; n2 = 5
      AA[ n1:n2,(n1+5):(n2+5) ] = C[:,:,0]
      AA[ n1:n2,n1:n2         ] = A[:,:,0]
      for n in range(1,N-1):
        n1 = 5*n; n2 = 5*(n+1)
        AA[ n1:n2, n1  :n2  ]     = A[:,:,n]
        AA[ n1:n2, (n1-5):(n2-5)] = B[:,:,n]
        AA[ n1:n2, (n1+5):(n2+5)] = C[:,:,n]
      n1 = 5*(N-1); n2 = 5*N
      AA[ n1:n2, n1    :n2     ] = A[:,:,-1];
      AA[ n1:n2, (n1-5):(n2-5) ] = B[:,:,-1];

      # eigenvalues of matrix
      om = np.linalg.eigvals(AA)
      kh=np.sqrt( kx[i]**2 + ky[j]**2 )
      # search minimal imaginary eigenvalue
      if kh>0: 
        om = np.extract( np.abs( np.real( om/kh )) <f_filter*cs, om )
        n=np.argmin( np.imag(om) )
        om_max[i,j]=om[n]
        # look for global minimum
        if np.imag(om[n]) < np.imag(omax):
           omax=om[n]; kmax=kx[i]; lmax=ky[j]; AAmax[:,:] = AA
  sys.stdout.write('\n')

  #eigenvectors for global minimum
  om, phi=np.linalg.eig(AAmax)
  n=np.argmin( np.imag(om) )
  om=om[n]
  phi = phi[:,n]

  # scale 
  A=2*np.pi*np.abs(np.imag(om))/(kmax**2+lmax**2)
  s=np.max( np.abs(np.real(phi[::5]))+np.abs(np.imag(phi[::5])) )
  phi=A*phi/s

  #complete solution
  str= phi[0::5]
  pot= phi[1::5]
  u= complex(0,1)*lmax*str-complex(0,1)*kmax*pot
  v=-complex(0,1)*kmax*str-complex(0,1)*lmax*pot
  w= phi[2::5]
  b= phi[3::5]
  p= phi[4::5]

  return om_max,om,kmax,lmax,u,v,w,b,p
 

def qg(U,V,B0,dz,kx,ky,beta,f0,hx,hy):
  import numpy as np
  import sys
  # quasi geostrophic vertical eigen value problem after
  # Smith (2007) The Geography of Linear Baroclinic 
  # Instability in Earth's Oceans, J. Mar. Res. 
  N=U.shape[0]
  # derivatives of U and B
  UZ=np.zeros(N,'d');VZ=np.zeros(N,'d');BZ=np.zeros(N,'d')
  for n in range(1,N-1):
   UZ[n]=(U[n+1]-U[n-1])/(2*dz)
   VZ[n]=(V[n+1]-V[n-1])/(2*dz)
   BZ[n]=(B0[n+1]-B0[n-1])/(2*dz)
  UZ[0]=UZ[1];UZ[-1]=UZ[-2]; BY=-f0*UZ
  VZ[0]=VZ[1];VZ[-1]=VZ[-2]; BX= f0*VZ
  BZ[0]=BZ[1];BZ[-1]=BZ[-2]

  # vertical differencing operator
  G      = np.zeros((N,N),'d')
  G[0,0] = -f0**2*( 1.0/(B0[1]-B0[0]))/dz
  G[0,1] = -f0**2*(-1.0/(B0[1]-B0[0]))/dz
  for n in range(1,N-1): 
    G[n,n-1] = -f0**2*(-1./(B0[n]-B0[n-1]))/dz
    G[n,n]   = -f0**2*( 1./(B0[n]-B0[n-1])+1./(B0[n+1]-B0[n]))/dz
    G[n,n+1] = -f0**2*(-1./(B0[n+1]-B0[n]))/dz
  G[-1,-1] = -f0**2*( 1./(B0[-1]-B0[-2]))/dz
  G[-1,-2] = -f0**2*(-1./(B0[-1]-B0[-2]))/dz

  # background PV gradient
  Qy = beta-np.dot(G,U) 
  Qx = np.dot(G,V)

  # topography
  Qx[-1]+=f0*hx/dz
  Qy[-1]+=f0*hy/dz


  om_max=np.zeros((kx.shape[0],ky.shape[0]),'Complex64')
  omax = complex(0,0); kmax=0.; lmax=0.
  Amax = np.zeros((N,N),'Complex64')

  # loop over wavenumbers
  for j in  range(ky.shape[0]):
    sys.stdout.write('\b'*22+' calculating j=%3i/%3i'%(j,ky.shape[0]) )
    sys.stdout.flush()
    for i in  range(kx.shape[0]):
      # construct matrixes and solve eigenvalue problem
      B = G-(kx[i]**2+ky[j]**2)*np.eye(N)
      A = kx[i]*np.diag(Qy)-ky[j]*np.diag(Qx) \
            +np.dot( kx[i]*np.diag(U)+ky[j]*np.diag(V) ,B)
      A = np.dot(np.linalg.inv(B),A) # this way is faster than gen. problem
      om= np.linalg.eigvals(A)
      n = np.argmin( np.imag(om) )
      om_max[i,j]=om[n]
      # look for global minimum
      if np.imag(om[n]) < np.imag(omax):
         omax=om[n]; kmax=kx[i]; lmax=ky[j]; Amax = A.copy()
  sys.stdout.write('\n')

  #eigenvector for global minimum
  om, phi = np.linalg.eig(Amax)
  n=np.argmin( np.imag(om) )
  om  = om[n]
  phi = phi[:,n]

  # scale 
  A=2*np.pi*np.abs(np.imag(om))/(kmax**2+lmax**2)
  s=np.max( np.abs(np.real(phi))+np.abs(np.imag(phi)) )
  phi=A*phi/s

  #complete solution
  u= complex(0,1.)*lmax*phi
  v=-complex(0,1.)*kmax*phi
  phiz=np.zeros( (N,), 'Complex64' )
  phiz[1:-1]= (phi[2:]-phi[:-2])/(2*dz)
  phiz[0]   = (phi[1]-phi[0])/dz
  phiz[-1]  = (phi[-1]-phi[-2])/dz
  b=f0*phiz
  p=f0*phi
  # N^2 w = - ( b_t + U b_x + V b_y - psi_y B_x + psi_x B_y)
  # w = -(i omega b - i k U b - i l V b + i l psi B_x - i k psi B_y)/BZ 
  w = -complex(0,1)*(om*b-kmax*U*b-lmax*V*b+lmax*phi*BX-kmax*phi*BY)/BZ 

  return om_max,om,kmax,lmax,u,v,w,b,p


