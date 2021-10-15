
import sys; sys.path.append('../../py_src')

from pyOM_gui import pyOM_gui as pyOM
#from pyOM_cdf import pyOM_cdf as pyOM

from numpy import *
from scipy.io.netcdf import netcdf_file as NF

class flame_E7(pyOM):
   """ the 4/3 FLAME Atlantic model
   """
   def set_parameter(self):
     """set main parameter
     """
     M=self.fortran.main_module   

     (M.nx,M.ny,M.nz)   = (87,89,45)
     M.dt_mom    = 3600.0
     M.dt_tracer = 3600.0

     M.runlen = 86400*365.0
     
     M.coord_degree     = 1

     M.congr_epsilon = 1e-6
     M.congr_max_iterations = 20000
     M.enable_streamfunction = 1

     I=self.fortran.isoneutral_module   
     I.enable_neutral_diffusion = 1
     I.enable_skew_diffusion = 1
     I.k_iso_0 = 1000.0
     I.k_iso_steep = 200.0
     I.iso_dslope = 1./1000.0
     I.iso_slopec = 4./1000.0

     M.enable_hor_friction = 1
     M.a_h = 5e4 
     M.enable_hor_friction_cos_scaling = 1
     M.hor_friction_cosPower = 3
     M.enable_tempsalt_sources = 1

     M.enable_implicit_vert_friction = 1
     T=self.fortran.tke_module   
     T.enable_tke = 1
     T.c_k = 0.1
     T.c_eps = 0.7
     T.alpha_tke = 30.0
     T.mxl_min = 1e-8
     T.tke_mxl_choice = 2

     M.k_gm_0 = 1000.0
     #E=self.fortran.eke_module   
     #E.enable_eke = 1

     I=self.fortran.idemix_module   
     I.enable_idemix = 1
     I.enable_idemix_hor_diffusion = 1

     M.eq_of_state_type = 5
     return


   def set_grid(self):
       M=self.fortran.main_module   
       fid = NF('forcing.cdf','r')
       (i1,i2) = ( self.if2py(M.is_pe), self.if2py(M.ie_pe) )
       M.dxt[i1:i2+1] = fid.variables['dxtdeg'][M.is_pe-1:M.ie_pe]
       (j1,j2) = ( self.jf2py(M.js_pe), self.jf2py(M.je_pe) )
       M.dyt[j1:j2+1] = fid.variables['dytdeg'][M.js_pe-1:M.je_pe] 
       M.dzt[:] = fid.variables['dzt'][::-1] /100.0
       M.x_origin = fid.variables['xu'][0]
       M.y_origin = fid.variables['yu'][0]
       return


   def set_coriolis(self):
     M=self.fortran.main_module
     for j in range( M.yt.shape[0] ): M.coriolis_t[:,j] = 2*M.omega*sin(M.yt[j]/180.*pi)
     return

   def set_topography(self):
       """ setup topography
       """
       M=self.fortran.main_module   
       fid = NF('forcing.cdf','r')
       kmt = fid.variables['kmt'][:]
       M.kbot[:]=1
       for j in range(M.js_pe,M.je_pe+1):
         for i in range(M.is_pe,M.ie_pe+1):
           (ii,jj) = ( self.if2py(i), self.jf2py(j) )
           M.kbot[ii,jj]=min(M.nz,M.nz-kmt[j-1,i-1]+1)
           if  kmt[j-1,i-1] == 0: M.kbot[ii,jj] =0
       return
   
   def set_initial_conditions(self):
       """ setup initial conditions
       """
       M=self.fortran.main_module   
       fid = NF('forcing.cdf','r')
       
       # initial conditions
       t = fid.variables['temp_ic'][::-1,M.js_pe-1:M.je_pe,M.is_pe-1:M.ie_pe]
       s = fid.variables['salt_ic'][::-1,M.js_pe-1:M.je_pe,M.is_pe-1:M.ie_pe]
       for k in range(M.nz):
          M.temp[ self.if2py(M.is_pe):self.if2py(M.ie_pe)+1, self.jf2py(M.js_pe):self.jf2py(M.je_pe)+1, k,M.tau-1] = t[k,:,:].transpose()
          M.salt[ self.if2py(M.is_pe):self.if2py(M.ie_pe)+1, self.jf2py(M.js_pe):self.jf2py(M.je_pe)+1, k,M.tau-1] = s[k,:,:].transpose()*1000+35.0
       for d in (M.temp,M.salt):
           d[...,M.tau-1] = d[...,M.tau-1]*M.maskt
           d[...,M.taum1-1] = d[...,M.tau-1]

       # wind stress
       self.tx = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, 12 ), 'd' , order = 'F')
       self.ty = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, 12 ), 'd' , order = 'F')
       for k in range(12):
           self.tx[2:-2,2:-2,k] = fid.variables['taux'][k,M.js_pe-1:M.je_pe,M.is_pe-1:M.ie_pe].transpose()/10. /M.rho_0
           self.ty[2:-2,2:-2,k] = fid.variables['tauy'][k,M.js_pe-1:M.je_pe,M.is_pe-1:M.ie_pe].transpose()/10. /M.rho_0
       # zero out masks and do boundary exchange
       for d in (self.tx, self.ty) :
         d[ d <= -1e10 ] = 0.0
         for k in range(12):
           self.fortran.border_exchg_xy(M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,d[:,:,k]) 
           self.fortran.setcyclic_xy   (M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,d[:,:,k])
       
       # interpolate from B grid location to C grid
       for k in range(12):
           self.tx[:,1:,k] = (self.tx[:,:-1,k] + self.tx[:,1:,k])/(M.maskz[:,:-1,-1]+M.maskz[:,1:,-1]+1e-20)
           self.ty[1:,:,k] = (self.ty[:-1,:,k] + self.ty[1:,:,k])/(M.maskz[:-1,:,-1]+M.maskz[1:,:,-1]+1e-20)
           self.tx[:,:,k] = self.tx[:,:,k]*M.masku[:,:,-1]
           self.ty[:,:,k] = self.ty[:,:,k]*M.maskv[:,:,-1]

       # heat flux and salinity restoring
       self.sst_clim = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, 12 ), 'd' , order = 'F')
       self.sss_clim = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, 12 ), 'd' , order = 'F')
       self.sst_rest = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, 12 ), 'd' , order = 'F')
       self.sss_rest = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, 12 ), 'd' , order = 'F')
       for k in range(12):
           self.sst_clim[2:-2,2:-2,k] = fid.variables['sst_clim'][k,M.js_pe-1:M.je_pe,M.is_pe-1:M.ie_pe].transpose()
           self.sss_clim[2:-2,2:-2,k] = fid.variables['sss_clim'][k,M.js_pe-1:M.je_pe,M.is_pe-1:M.ie_pe].transpose()*1000+35
           self.sst_rest[2:-2,2:-2,k] = fid.variables['sst_rest'][k,M.js_pe-1:M.je_pe,M.is_pe-1:M.ie_pe].transpose()*41868.
           self.sss_rest[2:-2,2:-2,k] = fid.variables['sss_rest'][k,M.js_pe-1:M.je_pe,M.is_pe-1:M.ie_pe].transpose()/100.0
       for d in (self.sst_clim, self.sss_clim, self.sst_rest, self.sss_rest): d[ d  <= -1e10 ] = 0.0

       # sponge layers
       self.t_star    = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, M.nz, 12 ), 'd' , order = 'F')
       self.s_star    = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, M.nz, 12 ), 'd' , order = 'F')
       self.rest_tscl = zeros( ( M.i_blk+2*M.onx, M.j_blk+2*M.onx, M.nz ), 'd' , order = 'F')
       fid = NF('restoring_zone.cdf','r')
       (i1,j1) = ( self.if2py(M.is_pe  ), self.jf2py(M.js_pe) )
       (i2,j2) = ( self.if2py(M.ie_pe+1), self.jf2py(M.je_pe+1) )
       for k in range(M.nz):
         self.rest_tscl[i1:i2,j1:j2,k] = fid.variables['tscl'][0,k,M.js_pe-1:M.je_pe,M.is_pe-1:M.ie_pe].transpose()
       for n in range(12):
         for k in range(M.nz):
           self.t_star[i1:i2,j1:j2,k,n] = fid.variables['t_star'][n,k,M.js_pe-1:M.je_pe,M.is_pe-1:M.ie_pe].transpose()
           self.s_star[i1:i2,j1:j2,k,n] = fid.variables['s_star'][n,k,M.js_pe-1:M.je_pe,M.is_pe-1:M.ie_pe].transpose()

       I=self.fortran.idemix_module   
       if I.enable_idemix:
         f = reshape( fromfile('tidal_energy.bin', dtype='>f'), (M.nx,M.ny), order='F' )/M.rho_0
         I.forc_iw_bottom[2:-2,2:-2]  = f[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe]
         f = reshape( fromfile('wind_energy.bin', dtype='>f'), (M.nx,M.ny), order='F' )/M.rho_0*0.2
         I.forc_iw_surface[2:-2,2:-2] = f[M.is_pe-1:M.ie_pe,M.js_pe-1:M.je_pe]
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
     fxa = 365*86400.0
     (f1,f2,n1,n2) = self.get_periodic_interval((M.itt-1)*M.dt_tracer,fxa,fxa/12.,12)
     
     M.surface_taux[:]=(f1*self.tx[:,:,n1-1] + f2*self.tx[:,:,n2-1])
     M.surface_tauy[:]=(f1*self.ty[:,:,n1-1] + f2*self.ty[:,:,n2-1])

     T=self.fortran.tke_module   
     if T.enable_tke:
       T.forc_tke_surface[1:-1,1:-1] = sqrt( (0.5*(M.surface_taux[1:-1,1:-1]+M.surface_taux[:-2,1:-1]) )**2  \
                                            +(0.5*(M.surface_tauy[1:-1,1:-1]+M.surface_tauy[1:-1,:-2]) )**2 )**(3./2.) 
     cp_0 = 3991.86795711963
     M.forc_temp_surface[:]=(f1*self.sst_rest[:,:,n1-1]+f2*self.sst_rest[:,:,n2-1])* \
                            (f1*self.sst_clim[:,:,n1-1]+f2*self.sst_clim[:,:,n2-1]-M.temp[:,:,-1,M.tau-1])*M.maskt[:,:,-1]/cp_0/M.rho_0
     M.forc_salt_surface[:]=(f1*self.sss_rest[:,:,n1-1]+f2*self.sss_rest[:,:,n2-1])* \
                            (f1*self.sss_clim[:,:,n1-1]+f2*self.sss_clim[:,:,n2-1]-M.salt[:,:,-1,M.tau-1])*M.maskt[:,:,-1]
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

if __name__ == "__main__": flame_E7().run(snapint= 86400.0)
