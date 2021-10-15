
import sys; sys.path.append('../py_src')
from pyOM_gui import pyOM_gui as pyOM
from numpy import *

BETA = 2e-11
F0   = 1e-4
CN   = 2.0
RN   = CN/F0

class rossby1(pyOM):
   """ Rossby waves
   """
   def set_parameter(self):
     """set main parameter
     """
     M=self.fortran.main_module   
     M.nx    = 32
     M.ny    = 32
     M.nz    = 1
     M.dt_tracer  = RN/CN*0.25
     M.dt_mom     = RN/CN*0.25

     M.congr_epsilon = 1e-12
     M.enable_streamfunction = 0
     M.enable_free_surface   = 1
     M.eq_of_state_type      = 1
     
     M.enable_conserve_energy = 0
     M.coord_degree           = 0
     
     M.enable_hydrostatic          = 1
     M.enable_cyclic_x             = 1
     M.enable_superbee_advection   = 1
     #M.enable_quicker_mom_advection= 0
     #M.enable_no_mom_advection     = 1
     #M.enable_biharmonic_friction  = 1
     return
 
   def set_grid(self):
     M=self.fortran.main_module
     print 'Rossby radius is ',RN/1e3,' km'
     M.dxt[:] = RN/10.0
     M.dyt[:] = RN/10.0
     M.dzt[:] = CN**2/9.81 
     #M.a_hbi  = 1e11*(M.dxt[0]/50e3)**4
     return

   def set_coriolis(self):
     """ vertical and horizontal Coriolis parameter on yt grid
         routine is called after initialization of grid
     """
     M=self.fortran.main_module   
     for j in range( M.yt.shape[0] ): M.coriolis_t[:,j]   =  F0+BETA*M.yt[j]
     return

   def set_initial_conditions(self):
     """ setup all initial conditions
     """
     M=self.fortran.main_module   
     y0=M.ny*M.dxt[0]*0.5
     L = RN/5
     kx = 2*pi/(M.nx*M.dxt[0])
     ky = pi/(M.ny*M.dyt[0])
     for i in range(M.xt.shape[0]):
       for j in range(M.yt.shape[0]):
         #M.psi[i,j,:]=0.01*exp( -(M.xt[i]-y0)**2/L**2 -(M.yt[j]-y0)**2/L**2 )
         M.psi[i,j,:]=0.01*cos( kx*M.xt[i] )
     return

   def make_plot(self):
     """ make a plot using methods of self.figure
     """
     if hasattr(self,'figure'):
       M=self.fortran.main_module         # fortran module with model variables
       x=M.xt[2:-2]/1e3
       y=M.yt[2:-2]/1e3

       self.figure.clf()
       ax=self.figure.add_subplot(111)
       a=M.psi[2:-2,2:-2,M.tau-1] 
       co=ax.contourf(x,y,a.transpose(),15)
       ax.set_xlabel('x [km]');
       ax.set_ylabel('y [km]');
       ax.set_title('$eta$ [m]');

       
       if M.itt>0:
         a=M.u[2:-2:2,2:-2:2,0,M.tau-1] 
         b=M.v[2:-2:2,2:-2:2,0,M.tau-1]
         ax.quiver(x[::2],y[::2],a.transpose(),b.transpose() )
       self.figure.colorbar(co)
     return

   
if __name__ == "__main__": 
   model= rossby1()
   model.run(snapint=0.5*86400.0,runlen=365*86400.)
