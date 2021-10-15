


from numpy import *
import pylab as plt



def pe(U,B0,dz,kx,fh,f0):
  # solution for primitive equations
  import numpy as np
  import sys
  cs = 150 # speed of sound, artificially reduced
  N=U.shape[0]
  # derivatives of U,V and B
  UZ=np.zeros(N,'d');BZ=np.zeros(N,'d')
  for n in range(1,N-1):
   UZ[n]=(U[n+1]-U[n-1])/(2*dz)
   BZ[n]=(B0[n+1]-B0[n-1])/(2*dz)
  UZ[0]=UZ[1];UZ[-1]=UZ[-2];
  BZ[0]=BZ[1];BZ[-1]=BZ[-2];
  # allocate some variables
  I=complex(0,1)
  A  = np.zeros((5,5,N),'Complex64')
  B  = np.zeros((5,5,N),'Complex64')
  C  = np.zeros((5,5,N),'Complex64')
  AA = np.zeros((5*N,5*N),'Complex64')
  om_max= np.zeros((kx.shape[0],),'Complex64');
  om_real= np.zeros((kx.shape[0],3*N),'float');
  omax  = complex(0,0)
  AAmax = np.zeros((5*N,5*N),'Complex64')
  kmax  = 0;
  # enter main loop
  for i in range(kx.shape[0]): # loop over zonal wavelength
      sys.stdout.write('\b'*21+'calculating i=%3i/%3i'%(i,kx.shape[0]) )
      sys.stdout.flush()

      Uk   = kx[i]*U # k \cdot \v U
      UZk  = kx[i]*UZ # k \cdot \v U'

      kh2 = kx[i]**2 + 1e-18

      for n in range(N): # loop over depth
        np1 = min(N-1,n+1)
        nm  = max(0,n-1)

        B[0,2,n]=  0.
        B[1,2,n]= -(kx[i]*fh+UZk[n])/(2*kh2)
        B[3,2,n]= I*BZ[n]/2
        B[4,2,n]= -cs**2*I/dz

        C[2,0,n]=  0
        C[2,1,n]= -kx[i]*fh/2
        C[2,3,n]= -I/2
        C[2,4,n]=  I/dz

        A[0,:,n]= [ Uk[n],I*f0,0,0,0 ]
        A[1,:,n]= [-I*f0 ,Uk[n], -(kx[i]*fh+UZk[n])/(2*kh2),0,I]
        A[2,:,n]= [ 0 ,-kx[i]*fh/2 ,(Uk[n]+Uk[np1])/2,-I/2,-I/dz ]
        A[3,:,n]= [ -f0*UZk[n],0,I*BZ[n]/2,Uk[n], 0  ]
        A[4,:,n]= [ 0, -I*cs**2*kh2 ,  I*cs**2/dz, 0   , 0 ]

      # upper boundary condition
      A[2,:,-1]=0; B[2,:,-1]=0; C[2,:,-1]=0;
      A[:,2,-1]=0
      C[:,2,-1]=0
      C[:,2,-2]=0

      # lower boundary condition
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
      kh=kx[i]
      # search minimal imaginary eigenvalue
      if kh>0:
        om = np.extract( np.abs( np.real( om/kh )) <0.5*cs, om )
        n=np.argmin( np.imag(om) )
        om_max[i]=om[n]
        om_real[i,:]=real(om)
        # look for global minimum
        if np.imag(om[n]) < np.imag(omax):
           omax=om[n]; kmax=kx[i]; AAmax[:,:] = AA
  sys.stdout.write('\n')

  #eigenvectors for global minimum
  om, phi=np.linalg.eig(AAmax)
  n=np.argmin( np.imag(om) )
  om=om[n]
  phi = phi[:,n]

  #complete solution
  str= phi[0::5]
  pot= phi[1::5]
  u= -complex(0,1)*kmax*pot
  v=-complex(0,1)*kmax*str
  w= phi[2::5]
  b= phi[3::5]
  p= phi[4::5]

  return om_max,om,kmax,u,v,w,b,p,om_real





N0 = 0.03
RI = 0.02
dz=0.25 /0.5
nz=40
dUdz = N0/RI**0.5               # Ri = N^2/Uz^2 ,   Uz = N/sqrt(Ri)
kx=linspace(0,2*pi/1.0,50);   
zw=arange(nz)*dz+ dz
zt=zw-dz/2.0

if 0:
 # EUC setup
 nz=40
 dz=7.5
 zw=arange(nz)*dz+ dz
 zt=zw-dz/2.0
 zt = zt-zw[-1]
 zw = zw-zw[-1]
 
 S=0.9*tanh((zt+25.0)/10.0)-1.35*tanh((zt+110.0)/15.0)+0.45*tanh((zt+225.0)/25.0)
 N2=0.2-0.5*tanh((zt+25.)/10.0)+0.3*tanh((zt+150.0)/25.0)
 U=zt*0
 B=zt*0
 for k in range(1,nz):
   B[k] = B[k-1] + N2[k]*dz
   U[k] = U[k-1] + 1.0*S[k]*dz
 T=-B/(-0.2)*1024/9.81
 kx=linspace(0,2*pi/2.,50);   
   
if 0:
 # piecewise linar profiles
 N0 = 0.0
 RI = 0.3
 U=dUdz*zt
 #U[nz/2:] = U[nz/2] 
 U[nz/2:] = U[nz/2:] + U[-1]/100
 B=N0**2*zt
 #B[nz/2+2:] = B[nz/2+2] 
 T=-B/(-0.2)*1024/9.81
 kx=linspace(0,1.0,50);   
 
if 0:
 # two layer flow
 T=-(9.85-6.5*tanh( (zt-zt[nz/2-1] ) /zt[0]*100 ))
 B=-T*(-0.2)*9.81/1024.
 U=0.6+0.5*tanh( (zt-zt[nz/2-1])/zt[0]*100)


if 0:
 # jet like
 u=1
 z1=3.
 S = N0/0.3**0.5               # Ri = N^2/Uz^2 ,   Uz = N/sqrt(Ri)
 U=0*zt
 for k in range(nz):
   if   zt[k]>z1: U[k]=-S*(zt[k]-z1)
   else:          U[k]=+S*(zt[k]-z1)
 T=N0**2*zt
 # kinks
 U[-6:] = U[-6] 
 #T[3*nz/4:] = T[3*nz/4] 
 B=-T*(-0.2)*9.81/1024.
 kx=linspace(0,0.2,40);   
 
if 1:
 # two vorticity jumps and one density jump
 dz=0.2 
 nz=50
 zw=arange(nz)*dz+ dz
 zt=zw-dz/2.0
 zt = zt - zw[-1]
 zw = zw - zw[-1]

 # vorticity jump
 u0=1.0
 z1=-6.0
 z2=-4.0
 S=2*u0/(z2-z1)
 print "S=", S
 U=0*zt-u0
 for k in range(nz):
   if   zt[k]>z1 and zt[k]< z2: U[k]=S*(zt[k]-z1)-u0
   elif zt[k]>=z2: U[k]=u0
 # density jump    
 gs=1.0
 T0 = gs*1024/9.81/0.2
 #T0 = 5.
 #gs=9.81*abs(T0*(-0.2)) /1024. 
 print "gs=", gs
 z3=-5
 T=0*zt
 for k in range(nz):
    if zt[k]>z3: T[k]=T0
   
 B=-T*(-0.2)*9.81/1024.
 kx=linspace(0,2.0,50);   
     
om_max,om,kmax,u,v,w,b,p,om_real=pe(U,B,dz,kx,0,0*2*(2*pi/86400.0))
print ' Max. growth rate PE %f 1/s ' % (-imag(om))
print ' k_max = %f ' % kmax


fig=plt.figure()
fig.clf()
ax=fig.add_subplot(221)
ax.plot(kx,-imag(om_max),color='k')

ax=fig.add_subplot(223)
for i in range(om_real.shape[1]):
  oo = om_real[:,i]
  for k in range(U.shape[0]):
    cond = logical_and( oo > U[k]*kx - 1e-3 , oo < U[k]*kx + 1e-3 )
    oo[ cond ] =  NaN
  for k in range(U.shape[0]-1):
    UU = -(U[k]+U[k+1])/2
    cond = logical_and( oo > UU*kx - 1e-3 , oo < UU*kx + 1e-3 )
    oo[ cond ] =  NaN
  om_real[:,i] = oo
ax.plot(kx,om_real,'k.')
oo=om_max
oo[ abs( imag(om_max)  ) < 1e-8] = NaN
ax.plot(kx,real(om_max),color='r',linewidth=3)
        

ax=fig.add_subplot(243)
ax.plot(T,zt,color='k')
ax.set_title('T')
ax=fig.add_subplot(244)
ax.plot(U,zt,'k')
ax.set_title('U')


ax=fig.add_subplot(247)
ax.plot(real(b),zt,color='k')
ax.plot(imag(b),zt,color='r')

ax=fig.add_subplot(248)
ax.plot(real(w),zt,'k--')
ax.plot(imag(w),zt,'r--')
ax.plot(real(u),zt,color='k')
ax.plot(imag(u),zt,color='r')

plt.show()

