from climate.pyom import PyOMLegacy

import numpy as np

yt_start = -39.0
yt_end = 43
yu_start = -40.0
yu_end = 42

class ACC2(PyOMLegacy):
   """
   A simple global model with a Southern Ocean and Atlantic part
   """
   def set_parameter(self):
     M=self.main_module

     (M.nx,M.ny,M.nz) = (30,42,15)
     M.dt_mom = 4800
     M.dt_tracer = 86400/2.0

     M.coord_degree = 1
     M.enable_cyclic_x = 1

     M.congr_epsilon = 1e-12
     M.congr_max_iterations = 5000
     M.enable_streamfunction = 1

     I=self.isoneutral_module
     I.enable_neutral_diffusion = 1
     I.k_iso_0 = 1000.0
     I.k_iso_steep = 500.0
     I.iso_dslope = 0.005
     I.iso_slopec = 0.01
     I.enable_skew_diffusion = 1

     M.enable_hor_friction = 1
     M.a_h = (2*M.degtom)**3*2e-11
     M.enable_hor_friction_cos_scaling = 1
     M.hor_friction_cospower=1

     M.enable_bottom_friction = 1
     M.r_bot = 1e-5

     M.enable_implicit_vert_friction = 1
     T=self.tke_module
     T.enable_tke = 1
     T.c_k = 0.1
     T.c_eps = 0.7
     T.alpha_tke = 30.0
     T.mxl_min = 1e-8
     T.tke_mxl_choice = 2
     #T.enable_tke_superbee_advection = 1

     M.k_gm_0 = 1000.0
     E=self.eke_module
     E.enable_eke = 1
     E.eke_k_max  = 1e4
     E.eke_c_k    = 0.4
     E.eke_c_eps  = 0.5
     E.eke_cross  = 2.
     E.eke_crhin  = 1.0
     E.eke_lmin   = 100.0
     E.enable_eke_superbee_advection = 1
     E.enable_eke_isopycnal_diffusion = 1

     I=self.idemix_module
     I.enable_idemix = 1
     I.enable_idemix_hor_diffusion = 1
     I.enable_eke_diss_surfbot = 1
     I.eke_diss_surfbot_frac = 0.2
     I.enable_idemix_superbee_advection = 1

     M.eq_of_state_type = 3

   def set_grid(self):
     M=self.main_module
     ddz = np.array([50.,70.,100.,140.,190.,240.,290.,340.,390.,440.,490.,540.,590.,640.,690.])
     M.dxt[:] = 2.0
     M.dyt[:] = 2.0
     M.x_origin=  0.0
     M.y_origin= -40.0
     M.dzt[:] = ddz[::-1]/2.5

   def set_coriolis(self):
     M=self.main_module
     M.coriolis_t[:,:] = 2*M.omega*np.sin(M.yt[None,:]/180.*np.pi)

   def set_topography(self):
     """ setup topography
     """
     M=self.main_module
     (X,Y)= np.meshgrid(M.xt,M.yt); X=X.transpose(); Y=Y.transpose()
     M.kbot[:] = 0
     M.kbot[X > 1.0] = 1
     M.kbot[Y < -20] = 1

   def set_initial_conditions(self):
     """ setup initial conditions
     """
     M=self.main_module
     XT, YT = np.meshgrid(M.xt,M.yt); XT=XT.transpose(); YT=YT.transpose()
     XU, YU = np.meshgrid(M.xu,M.yu); XU=XU.transpose(); YU=YU.transpose()

     # initial conditions
     M.temp[:,:,:,:] = ((1-M.zt[None,None,:]/M.zw[0])*15*M.maskT)[...,None]
     #M.temp[:,:,:,M.taum1] =  (1-M.zt[None,None,:]/M.zw[0])*15*M.maskT
     M.salt[:,:,:,:] = 35.0*M.maskT[...,None]
     #M.salt[:,:,:,M.taum1] = 35.0*M.maskT[:]

     # wind stress forcing
     for j in range(M.js_pe,M.je_pe+1):
          jj = self.jf2py(j)
          taux=0.0
          if  M.yt[jj]<-20 : taux =  .1e-3*np.sin(np.pi*(M.yu[jj]-yu_start)/(-20.0-yt_start))
          if  M.yt[jj]>10  : taux =  .1e-3*(1-np.cos(2*np.pi*(M.yu[jj]-10.0)/(yu_end-10.0)))
          M.surface_taux[:,jj] = taux*M.maskU[:,jj,-1]

     # surface heatflux forcing
     self.t_rest = np.zeros(M.u[:,:,1,0].shape)
     self.t_star = 15 * np.ones(M.u[:,:,1,0].shape)
     self.t_star[:,M.yt < -20.] = 15*(M.yt[M.yt < -20.] - yt_start)/(-20.0-yt_start)
     self.t_star[:,M.yt > 20.] = 15*(1-(M.yt[M.yt > 20.]-20)/(yt_end-20))
     self.t_rest = M.dzt[None,-1]/(30.*86400.)*M.maskT[:,:,-1]

     T=self.tke_module
     if T.enable_tke:
        T.forc_tke_surface[2:-2,2:-2] = np.sqrt((0.5*(M.surface_taux[2:-2,2:-2]+M.surface_taux[1:-3,2:-2]))**2  \
                                              +(0.5*(M.surface_tauy[2:-2,2:-2]+M.surface_tauy[2:-2,1:-3]))**2 )**(1.5)

     I=self.idemix_module
     if I.enable_idemix:
       I.forc_iw_bottom[:] =  1.0e-6*M.maskW[:,:,-1]
       I.forc_iw_surface[:] = 0.1e-6*M.maskW[:,:,-1]
     return


   def set_forcing(self):
     M=self.main_module
     M.forc_temp_surface[:] = self.t_rest*(self.t_star-M.temp[:,:,-1,1])
     return

   def set_diagnostics(self):
     M=self.main_module
     self.register_average(name='temp',long_name='Temperature',         units = 'deg C' , grid = 'TTT', var = lambda: M.temp[...,1])
     self.register_average(name='salt',long_name='Salinity',            units = 'g/kg' ,  grid = 'TTT', var = lambda: M.salt[...,1])
     self.register_average(name='u',   long_name='Zonal velocity',      units = 'm/s' ,   grid = 'UTT', var = lambda: M.u[...,1])
     self.register_average(name='v',   long_name='Meridional velocity', units = 'm/s' ,   grid = 'TUT', var = lambda: M.v[...,1])
     self.register_average(name='w',   long_name='Vertical velocity',   units = 'm/s' ,   grid = 'TTU', var = lambda: M.w[...,1])
     self.register_average(name='taux',long_name='wind stress',         units = 'm^2/s' , grid = 'UT',  var = lambda: M.surface_taux)
     self.register_average(name='tauy',long_name='wind stress',         units = 'm^2/s' , grid = 'TU',  var = lambda: M.surface_tauy)
     self.register_average(name='psi' ,long_name='Streamfunction',      units = 'm^3/s' , grid = 'UU',  var = lambda: M.psi[...,1])
     return

   def user_defined_signal(self):
       """ this routine must be called by all processors
       """
       M=self.main_module
       a = zeros( (M.nx,M.ny), 'd')
       a[M.is_pe-1:M.ie_pe,0] = M.xt[2:-2]
       self.fortran.pe0_recv_2d(a)
       self.xt_gl = a[:,0].copy()

       a[0,M.js_pe-1:M.je_pe] = M.yt[2:-2]
       self.fortran.pe0_recv_2d(a)
       self.yt_gl = a[0,:].copy()

       self.psi_gl = np.zeros( (M.nx,M.ny), 'd')
       self.psi_gl[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = np.where( M.maskz[2:-2,2:-2,-1] >0,  M.psi[2:-2,2:-2,M.tau] , NaN)
       self.fortran.pe0_recv_2d(self.psi_gl)

       self.temp_gl = np.zeros( (M.nx,M.ny,M.nz), 'd')
       for k in range(M.nz):
         a[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = np.where( M.maskT[2:-2,2:-2,k] >0,  M.temp[2:-2,2:-2,k,M.tau] , NaN)
         self.fortran.pe0_recv_2d(a)
         self.temp_gl[:,:,k]=a.copy()

       self.kappa_gl = np.zeros( (M.nx,M.ny,M.nz), 'd')
       for k in range(M.nz):
         a[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = np.where( M.maskW[2:-2,2:-2,k] >0,  M.kappah[2:-2,2:-2,k] , NaN)
         self.fortran.pe0_recv_2d(a)
         self.kappa_gl[:,:,k]=a.copy()

       return

   def make_plot(self):
       M=self.main_module

       self.set_signal('user_defined') # following routine is called by all PEs
       self.user_defined_signal()

       self.figure.clf()
       ax=self.figure.add_subplot(221)

       co=ax.contourf(self.yt_gl,M.zt,self.temp_gl[M.nx/2-1,:,:].transpose())
       self.figure.colorbar(co)
       ax.set_title('temperature')
       ax.set_ylabel('z [m]')
       ax.axis('tight')

       ax=self.figure.add_subplot(223)
       try:
        co=ax.contourf(self.yt_gl,M.zw,log10(self.kappa_gl[M.nx/2-1,:,:].transpose()) )
       except:
        pass
       self.figure.colorbar(co)
       ax.set_title('Diffusivity')
       ax.set_xlabel('Latitude [deg N]')
       ax.set_ylabel('z [m]')
       ax.axis('tight')

       ax=self.figure.add_subplot(122)
       co=ax.contourf(self.xt_gl,self.yt_gl,self.psi_gl.transpose()*1e-6)
       self.figure.colorbar(co)
       ax.set_title('Streamfunction [Sv]')
       ax.set_xlabel('Longitude [deg E]')
       ax.axis('tight')

if __name__ == "__main__":
    simulation = ACC2()
    simulation.setup()
    simulation.run()
