
import sys; sys.path.append('../../py_src')

from pyOM_gui import pyOM_gui as pyOM
#from pyOM_cdf import pyOM_cdf as pyOM

from numpy import *
from scipy.io.netcdf import netcdf_file as NF

class global_4deg(pyOM):
   """ global 4 deg model with 15 levels
   """
   def set_parameter(self):
     """set main parameter
     """
     M=self.fortran.main_module   

     (M.nx,M.ny,M.nz)    = (90,40,15)
     M.dt_mom    = 1800.0
     M.dt_tracer = 86400.0

     M.coord_degree     = 1
     M.enable_cyclic_x  = 1

     M.congr_epsilon = 1e-8
     M.congr_max_iterations = 20000
     M.enable_streamfunction = 1

     I=self.fortran.isoneutral_module   
     I.enable_neutral_diffusion = 1
     I.k_iso_0 = 1000.0
     I.k_iso_steep = 1000.0
     I.iso_dslope=4./1000.0
     I.iso_slopec=1./1000.0
     I.enable_skew_diffusion = 1

     M.enable_hor_friction  = 1; M.a_h = (4*M.degtom)**3*2e-11
     M.enable_hor_friction_cos_scaling = 1; M.hor_friction_cosPower=1
 
     M.enable_implicit_vert_friction = 1
     T=self.fortran.tke_module   
     T.enable_tke = 1
     T.c_k = 0.1
     T.c_eps = 0.7
     T.alpha_tke = 30.0
     T.mxl_min = 1e-8
     T.tke_mxl_choice = 2
     T.enable_tke_superbee_advection = 1

     E=self.fortran.eke_module   
     E.enable_eke = 1
     E.eke_k_max  = 1e4
     E.eke_c_k    = 0.4
     E.eke_c_eps  = 0.5
     E.eke_cross  = 2.
     E.eke_crhin  = 1.0
     E.eke_lmin   = 100.0
     E.enable_eke_superbee_advection = 1

     I=self.fortran.idemix_module   
     I.enable_idemix = 1
     I.enable_idemix_hor_diffusion = 1
     I.enable_eke_diss_surfbot = 1
     I.eke_diss_surfbot_frac = 0.2 # fraction which goes into bottom
     I.enable_idemix_superbee_advection = 1

     M.eq_of_state_type = 5
     return


   def set_grid(self):
       M=self.fortran.main_module   
       ddz = array([50.,70.,100.,140.,190.,240.,290.,340.,390.,440.,490.,540.,590.,640.,690.])
       M.dzt[:]=ddz[::-1]
       M.dxt[:] = 4.0
       M.dyt[:] = 4.0
       M.y_origin = -76.0
       M.x_origin = 4.0
       return


   def set_coriolis(self):
       M=self.fortran.main_module   
       for j in range( M.yt.shape[0] ): M.coriolis_t[:,j] = 2*M.omega*sin(M.yt[j]/180.*pi)
       return

   def set_topography(self):
         """ setup topography
         """
         M=self.fortran.main_module   
         M.kbot[:]=0
         bathy = reshape( fromfile('bathymetry.bin', dtype='>i'), (M.nx,M.ny), order='F' )
         lev_s = reshape( fromfile('lev_s.bin', dtype='>f', count=M.nx*M.ny*M.nz), (M.nx,M.ny,M.nz), order='F' )[:,:,::-1]
         for j in range(M.js_pe,M.je_pe+1):
           for i in range(M.is_pe,M.ie_pe+1):
              (ii,jj) = ( self.if2py(i), self.jf2py(j) )
              for k in range(M.nz-1,-1,-1): 
                  if lev_s[i-1,j-1,k] > 0.0: M.kbot[ii,jj]=k+1
              if bathy[i-1,j-1] == 0.0: M.kbot[ii,jj] =0
         M.kbot[ M.kbot == M.nz] = 0
         return
   
   def set_initial_conditions(self):
       """ setup initial conditions
       """
       M=self.fortran.main_module   

       self.taux = zeros( (M.i_blk+2*M.onx,M.j_blk+2*M.onx,12), 'd', order = 'F')
       self.tauy = zeros( (M.i_blk+2*M.onx,M.j_blk+2*M.onx,12), 'd', order = 'F')
       self.qnec = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, 12 ), 'd' , order = 'F')
       self.qnet = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, 12 ), 'd' , order = 'F')
       self.sst_clim = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, 12 ), 'd' , order = 'F')
       self.sss_clim = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, 12 ), 'd' , order = 'F')
       
       # initial conditions for T and S
       lev = reshape( fromfile('lev_t.bin', dtype='>f', count=M.nx*M.ny*M.nz), (M.nx,M.ny,M.nz), order='F' )
       lev = lev[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe,::-1]
       M.temp[M.onx:-M.onx,M.onx:-M.onx,:,M.tau-1] = lev*M.maskt[M.onx:-M.onx,M.onx:-M.onx,:]
       M.temp[...,M.taum1-1] = M.temp[...,M.tau-1]
       
       lev = reshape( fromfile('lev_s.bin', dtype='>f', count=M.nx*M.ny*M.nz), (M.nx,M.ny,M.nz), order='F' )
       lev = lev[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe,::-1]
       M.salt[M.onx:-M.onx,M.onx:-M.onx,:,M.tau-1] = lev*M.maskt[M.onx:-M.onx,M.onx:-M.onx,:]
       M.salt[...,M.taum1-1] = M.salt[...,M.tau-1]

       # use Trenberth wind stress from MITgcm instead of ECMWF (also contained in ecmwf_4deg.cdf)
       tx = reshape( fromfile('trenberth_taux.bin', dtype='>f'), (M.nx,M.ny,12), order='F' )/M.rho_0
       ty = reshape( fromfile('trenberth_tauy.bin', dtype='>f'), (M.nx,M.ny,12), order='F' )/M.rho_0
       self.taux[M.onx:-M.onx,M.onx:-M.onx,:] = tx[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe,:]
       self.tauy[M.onx:-M.onx,M.onx:-M.onx,:] = ty[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe,:]

       # heat flux 
       fid = NF('ecmwf_4deg_monthly.cdf','r')
       for k in range(12):
           self.qnec[M.onx:-M.onx,M.onx:-M.onx,k] = fid.variables['Q3'][k,M.js_pe-1:M.je_pe,M.is_pe-1:M.ie_pe].transpose()
       self.qnec[ self.qnec  <= -1e10 ] = 0.0

       q = reshape( fromfile('ncep_qnet.bin', dtype='>f'), (M.nx,M.ny,12), order='F' )
       self.qnet[M.onx:-M.onx,M.onx:-M.onx,:] = - q[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe,:]
       self.qnet[ self.qnet  <= -1e10 ] = 0.0

       fxa = zeros( (1,), 'd', order = 'F')
       fxb = zeros( (1,), 'd', order = 'F')
       for n in range(12):
           fxa = fxa + sum( self.qnet[M.onx:-M.onx,M.onx:-M.onx,n]*M.area_t[M.onx:-M.onx,M.onx:-M.onx] )
           fxb = fxb + sum( M.area_t[M.onx:-M.onx,M.onx:-M.onx] )
       self.fortran.global_sum(fxa)
       self.fortran.global_sum(fxb)
       fxa = fxa[0]/fxb[0]
       if M.my_pe==0: print  ' removing an annual mean heat flux imbalance of %e W/m^2'% fxa
       for n in range(12):
         self.qnet[:,:,n] = (self.qnet[:,:,n] - fxa)*M.maskt[:,:,-1]

       # SST and SSS
       sst = reshape( fromfile('lev_sst.bin', dtype='>f'), (M.nx,M.ny,12), order='F' )
       sss = reshape( fromfile('lev_sss.bin', dtype='>f'), (M.nx,M.ny,12), order='F' )
       self.sst_clim[M.onx:-M.onx,M.onx:-M.onx,:] = sst[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe,:]
       self.sss_clim[M.onx:-M.onx,M.onx:-M.onx,:] = sss[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe,:]

       I=self.fortran.idemix_module   
       if I.enable_idemix:
         f = reshape( fromfile('tidal_energy.bin', dtype='>f'), (M.nx,M.ny), order='F' )/M.rho_0
         I.forc_iw_bottom[ M.onx:-M.onx,M.onx:-M.onx]  = f[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe]
         f = reshape( fromfile('wind_energy.bin', dtype='>f'), (M.nx,M.ny), order='F' )/M.rho_0*0.2
         I.forc_iw_surface[M.onx:-M.onx,M.onx:-M.onx] = f[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe]
       return

   def get_periodic_interval(self,currentTime,cycleLength,recSpacing,nbRec):
       """  interpolation routine taken from mitgcm
       """
       locTime = currentTime - recSpacing*0.5 + cycleLength*( 2 - round(currentTime/cycleLength) )
       tmpTime = locTime % cycleLength 
       tRec1 = 1 + int( tmpTime/recSpacing )
       tRec2 = 1 + tRec1 % int(nbRec) 
       wght2 = ( tmpTime - recSpacing*(tRec1 - 1) )/recSpacing
       wght1 = 1.0 - wght2
       return (wght1,wght2,tRec1,tRec2)
   

   def set_forcing(self):
     M=self.fortran.main_module   
     
     (f1,f2,n1,n2) = self.get_periodic_interval((M.itt-1)*M.dt_tracer,365*86400.0,365*86400./12.,12)

     # wind stress
     M.surface_taux[:]=(f1*self.taux[:,:,n1-1] + f2*self.taux[:,:,n2-1])
     M.surface_tauy[:]=(f1*self.tauy[:,:,n1-1] + f2*self.tauy[:,:,n2-1])

     # tke flux
     T=self.fortran.tke_module   
     if T.enable_tke:
       T.forc_tke_surface[1:-1,1:-1] = sqrt( (0.5*(M.surface_taux[1:-1,1:-1]+M.surface_taux[:-2,1:-1]) )**2  \
                                            +(0.5*(M.surface_tauy[1:-1,1:-1]+M.surface_tauy[1:-1,:-2]) )**2 )**(3./2.) 
     # heat flux : W/m^2 K kg/J m^3/kg = K m/s 
     cp_0 = 3991.86795711963
     sst  =  f1*self.sst_clim[:,:,n1-1]+f2*self.sst_clim[:,:,n2-1] 
     qnec =  f1*self.qnec[:,:,n1-1]    +f2*self.qnec[:,:,n2-1]
     qnet =  f1*self.qnet[:,:,n1-1]    +f2*self.qnet[:,:,n2-1]
     M.forc_temp_surface[:] =(qnet+ qnec*(sst-M.temp[:,:,-1,M.tau-1]) )*M.maskt[:,:,-1]/cp_0/M.rho_0

     # salinity restoring
     t_rest= 30*86400.0
     sss  =  f1*self.sss_clim[:,:,n1-1]+f2*self.sss_clim[:,:,n2-1] 
     M.forc_salt_surface[:] =M.dzt[-1]/t_rest*(sss-M.salt[:,:,-1,M.tau-1])*M.maskt[:,:,-1]

     # apply simple ice mask                     
     n=nonzero( logical_and( M.temp[:,:,-1,M.tau-1]*M.maskt[:,:,-1] <= -1.8 , M.forc_temp_surface <= 0.0 ) )                            
     M.forc_temp_surface[n]=0.0
     M.forc_salt_surface[n]=0.0

     if M.enable_tempsalt_sources:
        M.temp_source[:] = M.maskt*self.rest_tscl*(f1*self.t_star[:,:,:,n1-1]+f2*self.t_star[:,:,:,n2-1] - M.temp[:,:,:,M.tau-1] )
        M.salt_source[:] = M.maskt*self.rest_tscl*(f1*self.s_star[:,:,:,n1-1]+f2*self.s_star[:,:,:,n2-1] - M.salt[:,:,:,M.tau-1] )
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
       self.psi_gl[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = where( M.maskz[2:-2,2:-2,-1] >0,  M.psi[2:-2,2:-2,M.tau-1] , NaN) 
       self.fortran.pe0_recv_2d(self.psi_gl)
       
       self.temp_gl = zeros( (M.nx,M.ny,M.nz), 'd', order = 'F')
       for k in range(M.nz):
         a[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = where( M.maskt[2:-2,2:-2,k] >0,  M.temp[2:-2,2:-2,k,M.tau-1] , NaN) 
         self.fortran.pe0_recv_2d(a)
         self.temp_gl[:,:,k]=a.copy()

       self.kappa_gl = zeros( (M.nx,M.ny,M.nz), 'd', order = 'F')
       for k in range(M.nz):
         a[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe] = where( M.maskw[2:-2,2:-2,k] >0,  M.kappah[2:-2,2:-2,k] , NaN) 
         self.fortran.pe0_recv_2d(a)
         self.kappa_gl[:,:,k]=a.copy()

       return

 

   def make_plot(self):
       M=self.fortran.main_module         # fortran module with model variables
       self.figure.clf()
       
       self.set_signal('user_defined') # following routine is called by all PEs
       self.user_defined_signal()
       
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

       ax=self.figure.add_subplot(222)
       co=ax.contourf(self.xt_gl,self.yt_gl,self.psi_gl.transpose()*1e-6)
       self.figure.colorbar(co)
       ax.set_title('Streamfunction [Sv]')
       ax.set_xlabel('Longitude [deg E]')
       ax.set_ylabel('Latitude [deg N]')
       ax.axis('tight')
       
       return

if __name__ == "__main__": global_4deg().run(snapint= 86400.0*25, runlen = 86400.*365*10)
