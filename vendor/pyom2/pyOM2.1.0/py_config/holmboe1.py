
import sys; sys.path.append('../py_src')

from pyOM_gui import pyOM_gui as pyOM
#from pyOM_cdf import pyOM_cdf as pyOM
from numpy import *


fac = 0.5
mix = 2e-5/fac**2

class kelv2(pyOM):
   """ 
   """
   def set_parameter(self):
     """set main parameter
     """
     M=self.fortran.main_module   

     (M.nx,M.ny,M.nz)=(int(64),  1, int(40) )
     M.dt_mom     = 0.05/fac
     M.dt_tracer  = 0.05/fac

     M.enable_conserve_energy = 0
     M.coord_degree           = 0
     M.enable_cyclic_x        = 1
     M.enable_hydrostatic     = 0
     M.eq_of_state_type       = 1 

     M.congr_epsilon = 1e-12
     M.congr_max_iterations = 5000
     M.congr_epsilon_non_hydro=   1e-8
     M.congr_max_itts_non_hydro = 5000    

     M.enable_explicit_vert_friction = 1
     M.kappam_0 = mix/fac**2
     M.enable_hor_friction = 1
     M.a_h = mix/fac**2

     M.enable_superbee_advection = 1
     return

   def set_mean_state1(self,dz):
     # two vorticity jumps and one density jump
     M=self.fortran.main_module   
     zw=arange(M.nz)*dz+ dz
     zt=zw-dz/2.0
     zt = zt - zw[-1]
     zw = zw - zw[-1]
     alpha = self.fortran.linear_eq_of_state.linear_eq_of_state_drhodt()
     u=1
     z1=-15.0
     z2=-5.0
     S=2*u/(z2-z1)
     self.U=0*zt-u
     for k in range(M.nz):
         if   zt[k]>z1 and zt[k]< z2: self.U[k]=S*(zt[k]-z1)-u
         elif zt[k]>=z2:              self.U[k]=u
     T0 = 1.0
     z3=-10.0
     self.T=0*zt
     for k in range(M.nz):
           if zt[k]>z3: self.T[k]=T0
     self.B=-self.T*alpha*9.81/1024.
     return


   def set_mean_state2(self,dz):
     # two layer flow
     M=self.fortran.main_module   
     zw=arange(M.nz)*dz+ dz
     zt=zw-dz/2.0
     zt = zt - zw[-1]
     zw = zw - zw[-1]
     alpha = self.fortran.linear_eq_of_state.linear_eq_of_state_drhodt()
     self.T=-(9.85-6.5*tanh( (zt-zt[M.nz/2-1] ) /zt[0]*10 ))
     self.B=-self.T*alpha*9.81/1024.
     self.U=0.6+0.5*tanh( (zt-zt[M.nz/2-1])/zt[0]*10)
     return 
 
   def calc_lin_stab(self,dz):
     M=self.fortran.main_module   
     alpha = self.fortran.linear_eq_of_state.linear_eq_of_state_drhodt()
     
     self.set_mean_state1(dz)
     #self.set_mean_state2(dz)
     
     k=linspace(0,1.0,50);   
     om_max,om,kmax,u,v,w,b,p=pe(self.U,self.B,dz,k,0,0)
     print ' Max. growth rate %f 1/s ' % (-imag(om))
     print ' k_max = %f ' % (kmax,)
     self.kmax = kmax
     self.u_pert = 5e-2*real(u)
     self.w_pert = 5e-2*real(w)
     self.T_pert = -5e-2*real(b)/alpha*1024/9.81
     return 
 

   def set_grid(self):
     M=self.fortran.main_module   
     M.dzt[:]=0.25/fac
     self.calc_lin_stab(0.25/fac)

     L = 2*2*pi/self.kmax    # two waves
     dx  =   L/M.nx 
     M.dt_tracer  = dx*0.02/0.25   # c = 0.25/0.05  dt =dx/c
     M.dt_mom     = dx*0.02/0.25
     print "dx=%f m, dt= %f s "%(dx,M.dt_tracer)
     
     M.dxt[:]=dx
     M.dyt[:]=dx
     return

         
   def set_initial_conditions(self):
     """ setup initial conditions
     """
     M=self.fortran.main_module
     for k in range(M.nz):
       for i in range(M.xt.shape[0]):
        M.temp[i,:,k,:] = self.T[k]+self.T_pert[k]*sin( self.kmax*M.xt[i] ) 
        M.u[i,:,k,:]    = self.U[k]+self.u_pert[k]*sin( self.kmax*M.xt[i] ) 
        M.w[i,:,k,:]    =           self.w_pert[k]*sin( self.kmax*M.xt[i] ) 
     return


   def user_defined_signal(self):
       """ this routine must be called by all processors
       """
       M=self.fortran.main_module  
       a = zeros( (M.nx,M.ny), 'd', order = 'F')
       a[M.is_pe-1:M.ie_pe,0] = M.xt[2:-2]
       self.fortran.pe0_recv_2d(a)
       self.xt_gl = a[:,0].copy()
       
       self.temp_gl = zeros( (M.nx,M.ny,M.nz), 'd', order = 'F')
       self.u_gl = zeros( (M.nx,M.ny,M.nz), 'd', order = 'F')
       self.w_gl = zeros( (M.nx,M.ny,M.nz), 'd', order = 'F')
       for k in range(M.nz):
         a[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = where( M.maskt[2:-2,2:-2,k] >0,  M.temp[2:-2,2:-2,k,M.tau-1] , NaN) 
         self.fortran.pe0_recv_2d(a)
         self.temp_gl[:,:,k]=a.copy()

         a[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = where( M.masku[2:-2,2:-2,k] >0,  M.u[2:-2,2:-2,k,M.tau-1] , NaN) 
         self.fortran.pe0_recv_2d(a)
         self.u_gl[:,:,k]=a.copy()

         a[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = where( M.maskw[2:-2,2:-2,k] >0,  M.w[2:-2,2:-2,k,M.tau-1] , NaN) 
         self.fortran.pe0_recv_2d(a)
         self.w_gl[:,:,k]=a.copy()

       return

   def make_plot(self):
       M=self.fortran.main_module
       
       self.set_signal('user_defined') # following routine is called by all PEs
       self.user_defined_signal()
       
       self.figure.clf()
       ax=self.figure.add_subplot(211)
       
       co=ax.contourf(self.xt_gl,M.zt, self.temp_gl[:,0,:].transpose())
       self.figure.colorbar(co)
       ax.quiver(self.xt_gl[::2],M.zt[::2],self.u_gl[::2,0,::2].transpose(),self.w_gl[::2,0,::2].transpose() )
       ax.set_title('Temperature [deg C]')
       ax.set_ylabel('z [m]')
       #ax.axis('tight')
       
       ax=self.figure.add_subplot(212)
       co=ax.contourf(self.xt_gl,M.zt, self.w_gl[:,0,:].transpose())
       self.figure.colorbar(co)
       ax.set_title('vertical velocity [m/s]')
       ax.set_xlabel('x [m]')
       return



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

  return om_max,om,kmax,u,v,w,b,p


if __name__ == "__main__":
      m=kelv2()
      dt=m.fortran.main_module.dt_tracer
      m.run( snapint = dt*100 ,runlen = dt*10000)
