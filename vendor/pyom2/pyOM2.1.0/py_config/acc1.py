
import sys; sys.path.append('../py_src')

from pyOM_gui import pyOM_gui as pyOM
#from pyOM_ave import pyOM_ave as pyOM

from numpy import *

hresol = 0.25
vresol = 1.0

class acc1(pyOM):
   """  idealised Southern Ocean, similar to Viebahn and Eden (2010) Ocean modeling
   """
   def set_parameter(self):
     """set main parameter
     """
     M=self.fortran.main_module   

     (M.nx,M.ny,M.nz)    = ( int(128*hresol),  int(128*hresol), int(18*vresol) )
     M.dt_mom    = 1200./hresol
     M.dt_tracer = 1200.0/hresol*5


     M.enable_conserve_energy = 0
     M.coord_degree           = 0
     M.enable_cyclic_x        = 1
     M.enable_hydrostatic     = 1
     M.eq_of_state_type       = 1
     
     M.congr_epsilon = 1e-6
     M.congr_max_iterations = 5000
     M.enable_streamfunction = 1
     
     M.enable_superbee_advection = 1

     M.enable_implicit_vert_friction = 1
     I=self.fortran.isoneutral_module   
     I.enable_TEM_friction = 1
     I.k_gm_0 = 1000.0

     M.enable_hor_friction  = 1
     M.a_h = 5e4 

     M.enable_biharmonic_friction  = 0 # for eddy resolving version
     M.a_hbi  = 5e11/hresol**2

     M.enable_bottom_friction = 1
     M.r_bot = 1e-5*vresol

     M.kappah_0=1.e-4/vresol
     M.kappam_0=1.e-3/vresol
     #M.a_h = (1./hresol)**2*5e4     # 1/T = A_1 /dx^2 = A_2 /(f dx)^2
     return


   def set_grid(self):
     M=self.fortran.main_module
     M.dxt[:] = 20e3/hresol
     M.dyt[:] = 20e3/hresol
     M.dzt[:] = 50.0/vresol
     return


   def set_coriolis(self):
     M=self.fortran.main_module   
     phi0 =- 25.0 /180. *pi
     betaloc = 2*M.omega*cos(phi0)/M.radius
     for j in range( M.yt.shape[0] ): M.coriolis_t[:,j] = 2*M.omega*sin(phi0)+betaloc*M.yt[j]
     return

   def set_topography(self):
     """ setup topography
     """
     M=self.fortran.main_module
     L_y = 0.0
     if M.my_blk_j == M.n_pes_j: L_y = M.yu[-(1+M.onx)]
     self.fortran.global_max(L_y)
     L_x = 0.0
     if M.my_blk_i == M.n_pes_i: L_x = M.xu[-(1+M.onx)]
     self.fortran.global_max(L_x)
     if M.my_pe==0: print ' domain size is ',L_x,' m x ',L_y,' m'
     self.L_x = L_x; self.L_y = L_y    
     (X,Y)= meshgrid(M.xt,M.yt); X=X.transpose(); Y=Y.transpose()
     M.kbot[:]=1
     M.kbot[ logical_and( Y>L_y*0.5 , logical_or(  X>L_x*0.75 ,  X<L_x*0.25 )  ) ]=0
     return

   def set_initial_conditions(self):
     M=self.fortran.main_module
     alpha = self.fortran.linear_eq_of_state.linear_eq_of_state_drhodt()
     (XT,YT)= meshgrid(M.xt,M.yt); XT=XT.transpose(); YT=YT.transpose()
     (XU,YU)= meshgrid(M.xu,M.yu); XU=XU.transpose(); YU=YU.transpose()

     n=YT < self.L_y*0.5
     M.surface_taux[n] = .1e-3*sin(2*pi*YU[n] /self.L_y)
     self.t_rest = M.dzt[-1]/(30.*86400)
     db=-30e-3 *M.rho_0/M.grav/alpha
     self.t_star = YT*0 + db
     n=YT < self.L_y*0.5
     self.t_star[n] = db*YT[n]/(self.L_y/2.0)
     n=YT > self.L_y*0.75
     self.t_star[n] = db*(1-(YT[n]-self.L_y*0.75)/(self.L_y*0.25) )
     return

 
   def set_forcing(self):
     M=self.fortran.main_module   
     M.forc_temp_surface[:]=self.t_rest*(self.t_star-M.temp[:,:,-1,M.tau-1])
     return

   def set_diagnostics(self):
     M=self.fortran.main_module   
     self.register_average(name='temp',long_name='Temperature',         units = 'deg C' , grid = 'TTT', var = M.temp)
     self.register_average(name='salt',long_name='Salinity',            units = 'g/kg' ,  grid = 'TTT', var = M.salt)
     self.register_average(name='u',   long_name='Zonal velocity',      units = 'm/s' ,   grid = 'UTT', var = M.u)
     self.register_average(name='v',   long_name='Meridional velocity', units = 'm/s' ,   grid = 'TUT', var = M.v)
     self.register_average(name='w',   long_name='Vertical velocity',   units = 'm/s' ,   grid = 'TTU', var = M.w)
     self.register_average(name='taux',long_name='wind stress',         units = 'm^2/s' , grid = 'UT',  var = M.surface_taux)
     self.register_average(name='tauy',long_name='wind stress',         units = 'm^2/s' , grid = 'TU',  var = M.surface_tauy)
     self.register_average(name='psi' ,long_name='Streamfunction',      units = 'm^3/s' , grid = 'UU',  var = M.psi)
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
       self.psi_gl[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] =  where( M.maskt[2:-2,2:-2,-1] >0,  M.psi[2:-2,2:-2,M.tau-1] , NaN) 

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

       return
   
   def make_plot(self):
       from scipy import stats
       M=self.fortran.main_module 
       
       self.set_signal('user_defined') # following routine is called by all PEs
       self.user_defined_signal()
       self.figure.clf()
       
       ax=self.figure.add_subplot(221)
       co=ax.contourf(self.xt_gl/1e3,self.yt_gl/1e3,self.temp_gl[:,:,-1].transpose())
       self.figure.colorbar(co)
       ax.set_title('temperature')
       
       ax=self.figure.add_subplot(222)
       co=ax.contourf(self.xt_gl/1e3,self.yt_gl/1e3,self.psi_gl.transpose()*1e-6)
       self.figure.colorbar(co)
       ax.set_title('Streamfunction [Sv]')
       
       ax=self.figure.add_subplot(223)
       co=ax.contourf(self.yt_gl/1e3,M.zt,stats.nanmean(self.u_gl,axis=0).transpose())
       self.figure.colorbar(co)
       ax.set_title('zonal velocity [m/s]')
       
       ax=self.figure.add_subplot(224)
       co=ax.contourf(self.yt_gl/1e3,M.zt,stats.nanmean(self.temp_gl,axis=0).transpose())
       self.figure.colorbar(co)
       ax.set_title('temperature [deg C]')
       
       return

if __name__ == "__main__": acc1().run(snapint= 86400*40.0, runlen = 365*86400.*200)
