
import sys; sys.path.append('../py_src')

from pyOM_gui import pyOM_gui as pyOM
#from pyOM_cdf import pyOM_cdf as pyOM
from numpy import *


HRESOLVE = 0.5
VRESOLVE = 0.5
N_0     = 0.004
M_0     = sqrt(1e-5*0.1/1024.0*9.801)
spg_width = int(3*HRESOLVE)
t_rest=1./(5.*86400)

#DX  = 32094.1729769
DX  = 30e3
Lx  = DX*128
H   = 1800.0

class jets1(pyOM):
   """ a wide channel model with zonal jets
   """
   def set_parameter(self):
     """set main parameter
     """
     M=self.fortran.main_module   
     (M.nx,M.ny,M.nz)  = ( 128*HRESOLVE , 128*HRESOLVE , 18*VRESOLVE )
     M.dt_tracer  = 1800.0/HRESOLVE
     M.dt_mom     = 1800.0/HRESOLVE

     M.enable_conserve_energy = 0
     M.coord_degree           = 0
     M.enable_cyclic_x        = 1
     M.enable_hydrostatic     = 1
     M.eq_of_state_type       = 1
     
     M.congr_epsilon = 1e-12
     M.congr_max_iterations = 5000
     M.enable_streamfunction = 1
     
     M.kappah_0=1e-4/VRESOLVE**2
     M.kappam_0=1e-4/VRESOLVE**2
     
     M.enable_ray_friction      = 1
     M.r_ray = 1e-7 
     
     #M.enable_hor_friction = 1
     #M.a_h = 100/HRESOLVE**2  
     M.enable_biharmonic_friction  = 1
     M.a_hbi  = 5e11/HRESOLVE**2
     
     M.enable_superbee_advection = 1
     M.enable_tempsalt_sources = 1
     return
 
   def set_grid(self):
     M=self.fortran.main_module
     M.dxt[:]  = Lx/M.nx
     M.dyt[:]  = Lx/M.ny
     M.dzt[:]  = H/M.nz
     return

   def set_coriolis(self):
     M=self.fortran.main_module   
     phi0 = 10. /180. *pi
     betaloc = 2*M.omega*cos(phi0)/M.radius
     for j in range( M.yt.shape[0] ):
       M.coriolis_t[:,j] = 2*M.omega*sin(phi0)+betaloc*M.yt[j]
     return

   def set_topography(self):
     """ setup topography
     """
     M=self.fortran.main_module   
     M.kbot[:]=1
     return

   def set_initial_conditions(self):
     """ setup all initial conditions
     """
     M=self.fortran.main_module   
     alpha = self.fortran.linear_eq_of_state.linear_eq_of_state_drhodt() # use here general eq. of state !!!!!!!!!!
     self.T_star = zeros( M.temp[:,:,:,0].shape, order = 'F' )
     self.T_rest = zeros( M.temp[:,:,:,0].shape, order = 'F' )

     for j in range(M.yt.shape[0]):
       for k in range(M.nz):
         fxa = 0.5e-3*sin(M.xt*8.5/Lx*pi) *1024./9.81 /alpha  # drho = drho/dt  dt
         self.T_star[:,j,k]  = (  32+ (M_0**2*M.yt[j] - N_0**2*M.zt[k])*1024./9.81 /alpha )*M.maskt[:,j,k] 
         M.temp[:,j,k,M.tau-1]     = (fxa+ self.T_star[:,j,k])*M.maskt[:,j,k]
     M.temp[:,:,:,M.taum1-1] = M.temp[:,:,:,M.tau-1]
       
     for j in range(1,spg_width+1):
       if j<=M.je_pe and j>= M.js_pe:
         jj = self.jf2py(j)
         self.T_rest[:,jj,:]  =  t_rest/j*M.maskt[:,jj,:]
     for j in range(M.ny-1,M.ny-spg_width-1,-1):
       if j<=M.je_pe and j>= M.js_pe:
         jj = self.jf2py(j)
         self.T_rest[:,jj+1,:]  =  t_rest/(M.ny-j) *M.maskt[:,jj+1,:]
     return
  

   def set_forcing(self):
     M=self.fortran.main_module   
     if M.enable_tempsalt_sources: M.temp_source[:]=self.T_rest*(self.T_star-M.temp[:,:,:,M.tau-1])*M.maskt
     return

   def user_defined_signal(self):
       """ this routine must be called by all processors
       """
       M=self.fortran.main_module  
       a = zeros( (M.nx,M.ny), 'd', order = 'F')
       a[M.is_pe-1:M.ie_pe,0] = M.xt[2:-2]
       self.fortran.pe0_recv_2d(a)
       self.xt_gl = a[:,0].copy()
       
       a[0,M.js_pe-1:M.je_pe] = M.yt[2:-2]
       self.fortran.pe0_recv_2d(a)
       self.yt_gl = a[0,:].copy()
       
       self.psi_gl = zeros( (M.nx,M.ny), 'd', order = 'F')
       self.psi_gl[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = where( M.maskz[2:-2,2:-2,-1] >0,  M.psi[2:-2,2:-2,M.tau-1] , NaN) 
       self.fortran.pe0_recv_2d(self.psi_gl)
       
       self.temp_gl = zeros( (M.nx,M.ny,M.nz), 'd', order = 'F')
       for k in range(M.nz):
         a[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = where( M.maskt[2:-2,2:-2,k] >0,  M.temp[2:-2,2:-2,k,M.tau-1] , NaN) 
         self.fortran.pe0_recv_2d(a)
         self.temp_gl[:,:,k]=a.copy()

       self.u_gl = zeros( (M.nx,M.ny,M.nz), 'd', order = 'F')
       for k in range(M.nz):
         a[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = where( M.masku[2:-2,2:-2,k] >0,  M.u[2:-2,2:-2,k,M.tau-1] , NaN) 
         self.fortran.pe0_recv_2d(a)
         self.u_gl[:,:,k]=a.copy()
        
       self.v_gl = zeros( (M.nx,M.ny,M.nz), 'd', order = 'F')
       for k in range(M.nz):
         a[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = where( M.maskv[2:-2,2:-2,k] >0,  M.v[2:-2,2:-2,k,M.tau-1] , NaN) 
         self.fortran.pe0_recv_2d(a)
         self.v_gl[:,:,k]=a.copy()
        
       return
   

   def make_plot(self):
     """ diagnose the model variables, could be replaced by other version
     """
     M=self.fortran.main_module         # fortran module with model variables
     self.set_signal('user_defined') # following routine is called by all PEs
     self.user_defined_signal()
     k=M.nz*3/4
     i=int(M.nx/2)
     self.figure.clf()
     
     ax=self.figure.add_subplot(221)
     co=ax.contourf(self.xt_gl/1e3,self.yt_gl/1e3,self.temp_gl[:,:,k].transpose())
     self.figure.colorbar(co)
     #ax.set_title('temperature')
     #ax.set_ylabel('y [km]')
     a=self.u_gl[::2,::2,k] 
     b=self.v_gl[::2,::2,k] 
     ax.quiver(self.xt_gl[::2]/1e3,self.yt_gl[::2]/1e3,a.transpose(),b.transpose() )
     
     ax=self.figure.add_subplot(222)
     co=ax.contourf(self.yt_gl/1e3,M.zt,self.temp_gl[i,:,:].transpose())
     self.figure.colorbar(co)
     #ax.set_title('temperature')
     
     ax=self.figure.add_subplot(223)
     co=ax.contourf(self.xt_gl/1e3,self.yt_gl/1e3,self.psi_gl.transpose()/1e6)
     self.figure.colorbar(co)
     #ax.set_title('[Sv]')
     #ax.set_ylabel('y [km]')
     
     ax=self.figure.add_subplot(224)
     co=ax.contourf(self.yt_gl/1e3,M.zt,self.u_gl[i,:,:].transpose())
     self.figure.colorbar(co)
     #ax.set_title('temperature')
     return

  
if __name__ == "__main__":
    model=jets1()
    model.run(snapint=3*86400.0,runlen=365*86400.)


