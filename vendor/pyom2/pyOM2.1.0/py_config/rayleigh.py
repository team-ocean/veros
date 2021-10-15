
import sys; sys.path.append('../py_src')
from numpy import *
from pyOM_gui import pyOM_gui as pyOM

fac = 1.0
mix = 2.5e-3

class rayleigh(pyOM):
    
   def set_parameter(self):
     """set main parameter
     """
     M=self.fortran.main_module   
     M.nx=int(2*32*fac)
     M.nz=int(20*fac)
     M.ny=1
     M.dt_tracer=0.25/fac
     M.dt_mom   =0.25/fac
     
     M.enable_conserve_energy = 0
     M.coord_degree           = 0
     M.enable_cyclic_x        = 1
     M.enable_hydrostatic     = 0
     
     M.eq_of_state_type       = 1 
     M.enable_tempsalt_sources = 1

     M.congr_epsilon = 1e-6
     M.congr_max_iterations = 5000
     M.congr_epsilon_non_hydro=   1e-6
     M.congr_max_itts_non_hydro = 5000    

     M.enable_explicit_vert_friction = 1
     M.kappam_0 = mix/fac**2
     M.enable_hor_friction = 1
     M.a_h = mix/fac**2

     M.enable_superbee_advection = 1
     #M.kappah_0 = mix/fac**2
     #M.enable_hor_diffusion = 1
     #M.k_h = mix/fac**2
     
     M.runlen =  86400.0
     return

   
   def set_grid(self):
     M=self.fortran.main_module   
     M.dxt[:]=0.5/fac 
     M.dyt[:]=0.5/fac 
     M.dzt[:]=0.5/fac 
     return
   
   def set_initial_conditions(self):
     """ setup all initial conditions
     """
     M=self.fortran.main_module   
     for i in range(M.is_pe,M.ie_pe+1): 
         ii = self.if2py(i)
         M.temp[ii,:,:,:] = 0.05*sin(M.xt[ii]/(20*M.dxt[2])*pi) 
     for k in [0,1,2]:     M.temp[:,:,:,k] = M.temp[:,:,:,k]*M.maskt
     return
 
   def set_forcing(self):
     """ setup all forcing
         surface and bottom boundary conditions
         might be variable in time, called every time step
     """
     M=self.fortran.main_module   
     M.temp_source[:,:,-1] = -175/4185.5 /M.dzt[-1]
     M.temp_source[:,:,0 ] =  175/4185.5 /M.dzt[0]
     return


   def make_plot(self):
     """ diagnose the model variables, could be replaced by other version
     """
     M=self.fortran.main_module         # fortran module with model variables
     self.figure.clf()
     ax=self.figure.add_subplot(211)
     a=where( M.maskt[2:-2,2,:] >0, M.temp[2:-2,2,:,M.tau-1], NaN)
     co=ax.contourf(M.xt[2:-2],M.zt,a.transpose(),arange(-2.5,2.5,0.25))
     self.figure.colorbar(co)
     ax.quiver(M.xt[2:-2:2],M.zt[::2],M.u[2:-2:2,2,::2,M.tau-1].transpose(),M.w[2:-2:2,2,::2,M.tau-1].transpose() )
     ax.set_title('temperature')
     ax.set_xlabel('x [m]')
     ax.set_ylabel('z [m]')
     ax.axis('tight')
     return

if __name__ == "__main__": rayleigh().run(5.0)

