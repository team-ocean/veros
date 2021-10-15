
import sys; sys.path.append('../py_src')


from pyOM_gui import pyOM_gui as pyOM
#from pyOM_cdf import pyOM_cdf as pyOM
from numpy import *

fac = 1.0
mix = 2.5e-3


class kelv1(pyOM):
   """ 
   """
   def set_parameter(self):
     """set main parameter
     """
     M=self.fortran.main_module   

     (M.nx,M.ny,M.nz)=(int(1.5*64*fac),  1, int(40*fac) )
     M.dt_mom     = 0.04/fac
     M.dt_tracer  = 0.04/fac

     M.enable_conserve_energy = 0
     M.coord_degree           = 0
     M.enable_cyclic_x        = 1
     M.enable_hydrostatic     = 0
     M.eq_of_state_type       = 1 

     M.congr_epsilon = 1e-12
     M.congr_max_iterations = 5000
     M.congr_epsilon_non_hydro=   1e-6
     M.congr_max_itts_non_hydro = 5000    

     M.enable_explicit_vert_friction = 1
     M.kappam_0 = mix/fac**2
     M.enable_hor_friction = 1
     M.a_h = mix/fac**2

     M.enable_tempsalt_sources = 1
     M.enable_momentum_sources = 1
     M.enable_superbee_advection = 1
     return


   def set_grid(self):
     M=self.fortran.main_module   
     M.dxt[:]=0.25/fac 
     M.dyt[:]=0.25/fac 
     M.dzt[:]=0.25/fac 
     return

   def t_star_fct(self,k):
     M=self.fortran.main_module
     return 9.85-6.5*tanh( (M.zt[k-1]-M.zt[M.nz/2-1] ) /M.zt[0]*100 )

   def u_star_fct(self,k):
     M=self.fortran.main_module   
     return 0.6+0.5*tanh( (M.zt[k-1]-M.zt[M.nz/2-1])/M.zt[0]*100)

         
   def set_initial_conditions(self):
     """ setup initial conditions
     """
     M=self.fortran.main_module   

     # target for restoring
     self.t_rest = zeros( M.u[:,:,:,0].shape, order = 'F' )
     self.t_star = zeros( M.u[:,:,:,0].shape, order = 'F' )
     self.u_star = zeros( M.u[:,:,:,0].shape, order = 'F' )
     
     for k in range(M.nz):
      self.t_star[:,:,k]=self.t_star_fct(k+1)
      self.u_star[:,:,k]=self.u_star_fct(k+1)
      for i in range(M.is_pe,M.ie_pe+1):
        if i < M.nx/8: self.t_rest[self.if2py(i),:,k] = 1./(15*M.dt_mom)

     # initial conditions
     from numpy.random import randn
     for k in range(M.nz):
       M.u[:,:,k,:]   = self.u_star_fct(k+1)
       for i in range(M.is_pe,M.ie_pe+1):
        ii = self.if2py(i)
        fxa=1e-3*M.zt[0]*sin(M.xt[ii]/(M.nx*M.dxt[1])*16*pi)*sin(M.zt[k]/( M.zt[0]-M.dzt[0]/2)*pi)
        M.temp[ii,:,k,:]= fxa+self.t_star_fct(k+1) 
     return
   

   def set_forcing(self):
     M=self.fortran.main_module   
     if M.enable_tempsalt_sources: M.temp_source[:]=self.t_rest*(self.t_star-M.temp[:,:,:,M.tau-1])#*M.maskt[:]
     if M.enable_momentum_sources: M.u_source[:]   =self.t_rest*(self.u_star-M.u[:,:,:,M.tau-1]   )#*M.masku[:]
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
       ax.set_xlabel('x [m]')
       ax.set_ylabel('z [m]')
       #ax.axis('tight')
       return


if __name__ == "__main__":
      m=kelv1()
      dt=m.fortran.main_module.dt_tracer
      m.run( snapint = 2.0 ,runlen = 50.0)
