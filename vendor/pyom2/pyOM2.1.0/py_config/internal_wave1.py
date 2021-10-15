
import sys; sys.path.append('../py_src')
from pyOM_gui import pyOM_gui as pyOM
from numpy import *

fac=1
N_0 = 2*pi/10.
OM0 = 1./(1.5*10)

class internal_wave1(pyOM):
   
   """ internal wave maker
   """
   
   def set_parameter(self):
     """set main parameter
     """
     M=self.fortran.main_module
     (M.nx,M.ny,M.nz) = (64*fac,1,64*fac)
     
     M.dt_mom   =0.025/fac
     M.dt_tracer=0.025/fac
     
     M.enable_conserve_energy = 0
     M.coord_degree           = 0
     M.enable_cyclic_x        = 1
     M.enable_hydrostatic     = 0
     M.eq_of_state_type       = 1

     M.congr_epsilon = 1e-12
     M.congr_max_iterations = 5000
     M.congr_epsilon_non_hydro=   1e-7
     M.congr_max_itts_non_hydro = 5000    

     M.enable_explicit_vert_friction = 1
     M.kappam_0 = 5e-3/fac**2   
     M.enable_hor_friction = 1
     M.a_h      = 5e-3/fac**2
     M.enable_superbee_advection = 1

     M.enable_tempsalt_sources = 1
     M.enable_momentum_sources = 1
     return

   def set_grid(self):
       M=self.fortran.main_module   
       M.dxt[:]= 0.25/fac
       M.dyt[:]= 0.25/fac
       M.dzt[:]= 0.25/fac
       return

   def set_topography(self):
       M=self.fortran.main_module   
       M.kbot[:]=0
       M.kbot[:,2:-2]=1
       return


   def set_initial_conditions(self):
     """ setup all initial conditions
     """
     M=self.fortran.main_module
     alpha = self.fortran.linear_eq_of_state.linear_eq_of_state_drhodt()
     grav = 9.81; rho0 = 1024.0

     self.t0  = zeros( (M.i_blk+2*M.onx,M.j_blk+2*M.onx,M.nz) , 'd', order='F')
     self.dt0 = zeros( (M.i_blk+2*M.onx,M.j_blk+2*M.onx,M.nz,3) , 'd', order='F')

     # prescribed background stratification
     for k in range(M.nz):
        self.t0[:,:,k]=-N_0**2*M.zt[k]/grav/alpha*rho0*M.maskt[:,:,k]
     return

   def set_forcing(self):
      M=self.fortran.main_module   
      # implement effect of background state 
      # update density, etc of last time step
      M.temp[:,:,:,M.tau-1] = M.temp[:,:,:,M.tau-1] + self.t0
      self.fortran.calc_eq_of_state(M.tau)
      M.temp[:,:,:,M.tau-1] = M.temp[:,:,:,M.tau-1] - self.t0
      
      # advection of background temperature
      self.fortran.advect_tracer(M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,self.t0,self.dt0[...,M.tau-1],M.nz) 
      M.temp_source[:] = (1.5+M.ab_eps)*self.dt0[...,M.tau-1] - ( 0.5+M.ab_eps)*self.dt0[...,M.taum1-1]

      i=M.nx/2
      if i >= M.is_pe and i<= M.ie_pe:
         ii = self.if2py(i); k=M.nz/2-1
         jj= self.jf2py(1)
         M.u_source[ii,jj,k]= M.masku[ii,jj,k]*1./(100*60.*M.dt_tracer)*sin(2*pi*OM0*M.itt*M.dt_tracer)
      return

  
   def make_plot(self):
       M=self.fortran.main_module   
       self.figure.clf()
       ax=self.figure.add_subplot(111)
       x=M.xt[2:-2];z=M.zt[:];
       jj= self.jf2py(1); ii = self.if2py(M.nx/2); k=M.nz/2-1
       a= M.temp[:,jj,:,M.tau-1]
       a[ii-2:ii+3,k-2:k+3]=0

       co=ax.contourf(x,z,a[2:-2,:].transpose() )
       self.figure.colorbar(co)
       a = a+self.t0[:,jj,:]
       ax.contour(x,z,a[2:-2,:].transpose(),10,colors='red')

       a= M.u[:,jj,:,M.tau-1]
       b= M.w[:,jj,:,M.tau-1]
       a[ii-3:ii+4,k-3:k+4]=0
       b[ii-3:ii+4,k-3:k+4]=0
       ax.quiver(x[::2],z[::2],a[2:-2:2,::2].transpose(),b[2:-2:2,::2].transpose() )
 
       ax.set_title('Temperature')
       ax.set_xlabel('x [m]')
       ax.set_ylabel('z [m]')
       ax.axis('tight')
       return
  
if __name__ == "__main__": 
     internal_wave1().run(snapint = 1.0 ,runlen = 50.0)

