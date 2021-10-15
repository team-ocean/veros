


import sys; sys.path.append('../py_src')
from pyOM_gui import pyOM_gui as pyOM
#from pyOM_cdf import pyOM_cdf as pyOM
from numpy import *

class eady1(pyOM):
   """ Eady (1941) solution
   """
   def set_parameter(self):
     """set main parameter
     """
     M=self.fortran.main_module   
     
     (M.nx,M.ny,M.nz)    = (32,32,20)
     M.dt_tracer      = 1200.0 
     M.dt_mom         = 1200.0 
     
     M.congr_epsilon = 1e-12
     M.congr_max_iterations = 5000
     M.enable_streamfunction = 1
     
     M.enable_hydrostatic      = 1
     M.enable_cyclic_x         = 1
     
     M.enable_superbee_advection  = 1
     M.enable_explicit_vert_friction = 1
     M.enable_hor_friction = 1
     M.a_h = (20e3)**3*2e-11    
     M.kappam_0   = 1.e-4 
     M.kappah_0   = 1.e-4 
     
     M.enable_conserve_energy = 0
     M.coord_degree            = 0
     M.eq_of_state_type        = 1 
     M.enable_tempsalt_sources = 1
     return


   def set_grid(self):
       M=self.fortran.main_module   
       M.dxt[:]= 20e3
       M.dyt[:]= 20e3
       M.dzt[:]= 100.0
       return

   def set_topography(self):
       M=self.fortran.main_module   
       M.kbot[:]=0
       M.kbot[:,2:-2]=1
       return

   
   def set_coriolis(self):
     M=self.fortran.main_module   
     for j in range( M.yt.shape[0] ): M.coriolis_t[:,j] = 1e-4+0e-11*M.yt[j]
     return

   def set_forcing(self):
      M=self.fortran.main_module   
      
      # update density, etc of last time step
      M.temp[:,:,:,M.tau-1] = M.temp[:,:,:,M.tau-1] + self.t0
      self.fortran.calc_eq_of_state(M.tau)
      M.temp[:,:,:,M.tau-1] = M.temp[:,:,:,M.tau-1] - self.t0
      
      # advection of background temperature
      self.fortran.advect_tracer(M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,self.t0,self.dt0[...,M.tau-1],M.nz) 
      M.temp_source[:] = (1.5+M.ab_eps)*self.dt0[...,M.tau-1] - ( 0.5+M.ab_eps)*self.dt0[...,M.taum1-1]
      return
     

   def set_initial_conditions(self):
     """ setup all initial conditions
     """
     M=self.fortran.main_module   
     U_0 = 0.5
     N_0 = 0.004
     f=M.coriolis_t[M.ny/2]
     h = (M.nz-2)*M.dzt[0]
     kx=1.6*f/(N_0*h)
     ky=pi/((M.ny-2)*M.dxt[0])
     d=f/N_0/(kx**2+ky**2)**0.5

     fxa=(exp(h/d)+exp(-h/d))/(exp(h/d)-exp(-h/d))
     c1= (1+0.25*(h/d)**2-h/d*fxa )*complex(1,0)
     c1=(sqrt(c1)*d/h+0.5)*U_0
     A=(U_0-c1)/U_0*h/d
     
     alpha = self.fortran.linear_eq_of_state.linear_eq_of_state_drhodt()
     grav = 9.81; rho0 = 1024.0
        
     # zonal velocity 
     for k in range(M.nz):
       M.u[:,:,k,M.tau-1]= (U_0/2+U_0*M.zt[k]/(M.nz*M.dzt[0]))*M.masku[:,:,k] 
     M.u[...,M.taum1-1] = M.u[...,M.tau-1]
     
     self.t0  = zeros( (M.i_blk+2*M.onx,M.j_blk+2*M.onx,M.nz) , 'd', order='F')
     self.dt0 = zeros( (M.i_blk+2*M.onx,M.j_blk+2*M.onx,M.nz,3) , 'd', order='F')

     # rho = alpha T ,  N^2 = b_z = - g/rho0 rho_z = - g/rho0 alpha T_z,  T = - N^2 z rho0/(g alpha)
     for k in range(M.nz):
        self.t0[:,:,k]=-N_0**2*M.zt[k]/grav/alpha*rho0*M.maskt[:,:,k]
     
     # fu = -p_y, p_z = -g rho,  f u_z = -g rho_y,  rho_y = - f u_z/g = alpha T_y
     for k in range(M.nz):
      for j in range(1,M.ny+1):
        jj = self.jf2py(j)
        uz = U_0/M.ht[:,jj]
        self.t0[:,jj+1,k]=(self.t0[:,jj,k]+M.dyt[jj]*uz*f/grav/alpha*rho0)*M.maskt[:,jj,k]
        
        
     # perturbation buoyancy
     for k in range(M.nz):
       for j in range(1,M.ny+1):
        jj = self.jf2py(j)
        phiz=A/d*sinh(M.zt[k]/d)+cosh(M.zt[k]/d)/d
        M.temp[:,jj,k,M.tau-1] =0.1*sin(kx*M.xt)*sin(ky*M.yt[jj])*abs(phiz)*M.maskt[:,jj,k]*rho0/grav/alpha
        #M.temp[:,jj,k,M.tau-1] =1e-5*random.randn(M.nx+2*M.onx)*M.maskt[:,jj,k]
     M.temp[...,M.taum1-1] = M.temp[...,M.tau-1]
     return


   def make_plot(self):
     """ make a plot using methods of self.figure
     """
     if hasattr(self,'figure'):
       M=self.fortran.main_module         # fortran module with model variables
       k=M.nz*3/4
       i=self.if2py(M.nx/2)
       j=self.jf2py(M.ny/2)
       x=M.xt[2:-2]/1e3
       y=M.yt[2:-2]/1e3
       z=M.zt[:]

       self.figure.clf()
       ax=self.figure.add_subplot(221)
       a=M.temp[i,2:-2,:,M.tau-1] +self.t0[i,2:-2,:]
       co=ax.contourf(y,z,a.transpose())
       ax.set_ylabel('y [km]')
       self.figure.colorbar(co)
       a=M.u[i,2:-2,:,M.tau-1]
       ax.contour(y,z,a.transpose(),10,colors='black')
       ax.set_title('total Temperature');  
       ax.axis('tight') 
         
              
       
       ax=self.figure.add_subplot(222)
       a= M.temp[2:-2,2:-2,k,M.tau-1]  
       co=ax.contourf(x,y,a.transpose())
       self.figure.colorbar(co)
       a=M.temp[2:-2,2:-2,k,M.tau-1] +self.t0[2:-2,2:-2,k]
       ax.contour(x,y,a.transpose(),10,colors='black')
       ax.axis('tight')

                       
       
       ax=self.figure.add_subplot(223)
       a=M.v[2:-2,j,:,M.tau-1] 
       co=ax.contourf(x,z,a.transpose())
       ax.set_title('meridional velocity'); 
       ax.set_xlabel('x [km]'); 
       self.figure.colorbar(co)
       ax.axis('tight')
       
       
       ax=self.figure.add_subplot(224)
       a=M.temp[2:-2,j,:,M.tau-1] 
       co=ax.contourf(x,z,a.transpose())
       ax.set_title('temperature perturbation'); 
       ax.set_xlabel('x [km]'); 
       self.figure.colorbar(co)
       ax.axis('tight')
       
       
     return

if __name__ == "__main__": 
    model = eady1()
    M=model.fortran.main_module
    model.run( snapint = 86400.0)




