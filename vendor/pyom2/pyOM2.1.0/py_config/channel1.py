
import sys; sys.path.append('../py_src')
from pyOM_gui import pyOM_gui as pyOM
from numpy import *
import lin_stab

GRID_POINTS = 32      # number of grid points in lateral direction
EDDY_LENGTH_SCALES = 2 # number of eddies in domain

if 0: # meso-scale regime
  Ro  = 0.05
  Ek  = 0.001
  CFL = 0.02    # CFL number
  delta = 0.01 # aspect ratio

if 1: # quite sub-meso-scale regime
  Ro  = 0.1
  Ek  = 0.001
  CFL = 0.05   # CFL number
  delta = 0.02 # aspect ratio
if 0: # very sub-meso-scale regime
  Ro  = 0.5
  Ek  = 0.01   # without quicker mom adv, else 0.01
  CFL = 0.05   # CFL number
  delta = 0.01 # aspect ratio

H0 = 500.0
beta= 0e-11  # beta 
f0  = 1e-4   # coriolis freq.
N0=f0/delta  # stability freq.
U0 = Ro*N0*H0


class channel1(pyOM):
   """ channel with restoring zones
   """
   def set_parameter(self):
     """set main parameter
     """
     M=self.fortran.main_module   
     M.nx    = GRID_POINTS
     M.ny    = GRID_POINTS
     M.nz    = 18
     M.enable_superbee_advection    = 1
     #M.enable_hor_friction = 1
     M.enable_biharmonic_friction = 1
     
     M.enable_hydrostatic    = 1
     M.enable_cyclic_x       = 1
     M.enable_streamfunction = 1
     M.enable_conserve_energy = 0
     M.coord_degree           = 0
     M.eq_of_state_type       = 1
     M.enable_tempsalt_sources = 1
     return
 
   def set_grid(self):
     M=self.fortran.main_module   
     # print some numbers
     Lr    = N0*H0/f0          # Rossby radius
     Ro    = U0/(f0*Lr)       # Rossby number
     Ri    = N0**2*H0**2/U0**2 # Richardson number
     print
     print  ' L  = %f km'%(Lr/1e3)
     print  ' Ro = %f '%Ro
     print  ' Ri = %f '%Ri
     print  ' delta = %f '%delta
     print  ' ell = %f '%(Lr/6400e3)

     # solve linear stability problem first
     ksx=linspace(0,3.2,50);    kx=ksx/Lr
     ky = array( [0./Lr] )
     #ksy=linspace(-3.2,3.2,30); ky=ksy/Lr
     M.dzt[:]    = H0/M.nz
     zw=arange(M.nz)*M.dzt[0]+ M.dzt[0]
     zt=zw-M.dzt[0]/2.0
     zt = zt - zw[-1]
     zw = zw - zw[-1]
     U=U0/2+U0*zt/H0
     V=U*0
     B=N0**2*zt
     om_max,om,kmax,lmax,u,v,w,b,p=lin_stab.pe(U,V,B,M.dzt[0],kx,ky,0.,beta,0.,f0,0.,0)
     print ' Max. growth rate %f 1/days ' % (-imag(om)*86400)
     print ' k_max = %f Lr , l_max = %f Lr' % (kmax*Lr,lmax*Lr)

     self.lin_stab_om = om
     self.lin_stab_kmax = kmax
     self.lin_stab_b, self.lin_stab_u, self.lin_stab_v, self.lin_stab_w, self.lin_stab_p = b,u,v,w,p
     L = EDDY_LENGTH_SCALES*2*pi/kmax   
     M.dxt[:]  =   L/M.nx 
     M.dyt[:]  =   L/M.ny 
     M.dt_mom    = CFL/U0*M.dxt[0]   # CFL=U*dt/dx
     M.dt_tracer = CFL/U0*M.dxt[0]
     print " dx=%f km, dt= %f s "%(M.dxt[0]/1e3,M.dt_mom)
     print " CFL  = ",U0*M.dt_mom/M.dxt[0]
     print " CFL  = ",real(om)/kmax*M.dt_mom/M.dxt[0]
     M.congr_epsilon = 1e-12 *(M.dxt[0]/20e3)**2

     M.a_h      = Ek*f0*M.dxt[0]**2 
     M.a_hbi    = Ek*f0*M.dxt[0]**4 
     #M.kappam_0 = Ek*f0*M.dzt[0]**2
     
     #M.k_h = Ek*f0*M.dxt[0]**2 
     #M.k_v = Ek*f0*M.dzt[0]**2 
     print " A_h = %f m^2/s  Ek = %f"%(M.a_h,Ek)
     print " A_v = %f m^2/s  Ek = %f"%(M.kappam_0,Ek)
     return

   def set_coriolis(self):
     M=self.fortran.main_module   
     for j in range( M.yt.shape[0] ): M.coriolis_t[:,j] = f0+beta*M.yt[j]
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
     alpha = self.fortran.linear_eq_of_state.linear_eq_of_state_drhodt()
 
     # background velocity
     for k in range(M.nz):
        M.u[:,:,k,M.tau-1]=(U0/2+U0*M.zt[k]/H0)*M.masku[:,:,k]

     # background buoyancy
     self.t0  = zeros( (M.i_blk+2*M.onx,M.j_blk+2*M.onx,M.nz) , 'd', order='F')
     self.dt0 = zeros( (M.i_blk+2*M.onx,M.j_blk+2*M.onx,M.nz,3) , 'd', order='F')
     
     for k in range(M.nz):
        self.t0[:,:,k]=-N0**2*M.zt[k]/M.grav/alpha*M.rho_0*M.maskt[:,:,k]
        
     for k in range(M.nz):
      for j in range(1,M.ny+1):
        jj = self.jf2py(j)
        f   = (M.coriolis_t[0,jj]+M.coriolis_t[0,jj+1])/2.0
        uz = U0/H0
        self.t0[:,jj+1,k]=(self.t0[:,jj,k]+M.dyt[jj]*uz*f/M.grav/alpha*M.rho_0)*M.maskt[:,jj,k]

     # perturbation buoyancy, etc
     for k in range(M.nz):
      for j in range(M.yt.shape[0]):
        ky=pi/(M.ny*M.dyt[0])
        kx = self.lin_stab_kmax
        om = self.lin_stab_om
        amp = 0.2
        
        phase = cos(kx*M.xt) +complex(0,1)*sin(kx*M.xt)
        phase = phase*sin(ky*M.yt[j])
        M.temp[:,j,k,M.tau-1] =amp*real(phase*self.lin_stab_b[k])*M.maskt[:,j,k]*M.rho_0/(M.grav*alpha)
        M.u[:,j,k,M.tau-1] = M.u[:,j,k,M.tau-1] + amp*real(phase*self.lin_stab_u[k])*M.masku[:,j,k]
        M.v[:,j,k,M.tau-1] =                      amp*real(phase*self.lin_stab_v[k])*M.maskv[:,j,k]
        M.w[:,j,k,M.tau-1] =                      amp*real(phase*self.lin_stab_w[k])*M.maskw[:,j,k]
        
        phase = cos(kx*M.xt+real(om)*M.dt_tracer) +complex(0,1)*sin(kx*M.xt+real(om)*M.dt_tracer)
        phase = phase*sin(ky*M.yt[j])
        M.temp[:,j,k,M.taum1-1] =amp*real(phase*self.lin_stab_b[k])*M.maskt[:,j,k]*M.rho_0/(M.grav*alpha)
        M.u[:,j,k,M.taum1-1] = M.u[:,j,k,M.tau-1] + amp*real(phase*self.lin_stab_u[k])*M.masku[:,j,k]
        M.v[:,j,k,M.taum1-1] =                      amp*real(phase*self.lin_stab_v[k])*M.maskv[:,j,k]
        M.w[:,j,k,M.taum1-1] =                      amp*real(phase*self.lin_stab_w[k])*M.maskw[:,j,k]

     M.du[:,:,:,M.taum1-1] = (M.u[:,:,:,M.tau-1]-M.u[:,:,:,M.taum1-1] )/M.dt_mom
     M.dv[:,:,:,M.taum1-1] = (M.v[:,:,:,M.tau-1]-M.v[:,:,:,M.taum1-1] )/M.dt_mom
     M.dtemp[:,:,:,M.taum1-1] = (M.temp[:,:,:,M.tau-1]-M.temp[:,:,:,M.taum1-1] )/M.dt_tracer
     #for d in (M.temp,M.u,M.v,M.w): d[...,M.taum1-1] = d[...,M.tau-1]

     return
 

   def make_plot(self):
     """ make a plot using methods of self.figure
     """
     if hasattr(self,'figure'):
       M=self.fortran.main_module   
       k=M.nz*3/4
       i=int(M.nx/2)
       j=int(M.nx/2)
       x=M.xt[2:-2]/1e3
       y=M.yt[2:-2]/1e3
       z=M.zt[:]

       self.figure.clf()
       if 1:
         ax=self.figure.add_subplot(221)
         a=M.temp[i,2:-2,:,M.tau-1] + self.t0[i,2:-2,:]
         co=ax.contourf(y,z,a.transpose())
         a=M.u[i,2:-2,:,M.tau-1] 
         ax.contour(y,z,a.transpose(),10,colors='black')
         ax.set_ylabel('y [km]')
         self.figure.colorbar(co)
         ax.set_title('total temperature');  
         ax.axis('tight') 
         
       ax=self.figure.add_subplot(222)
       co=ax.contourf(x,y,M.temp[2:-2,2:-2,k,M.tau-1].transpose())
       a=M.temp[2:-2,2:-2,k,M.tau-1] + self.t0[2:-2,2:-2,k]
       ax.contour(x,y,a.transpose(),10,colors='black')
       ax.set_title('temperature perturbation'); 
       self.figure.colorbar(co)
       ax.axis('tight')

           
       ax=self.figure.add_subplot(223)
       a=M.temp[2:-2,j,:,M.tau-1] 
       co=ax.contourf(x,z,a.transpose())
       ax.set_title("temperature perturb. at y=%4.1f km"%y[j]); 
       ax.set_xlabel('x [km]'); 
       self.figure.colorbar(co)
       ax.axis('tight')

       ax=self.figure.add_subplot(224)
       a=M.temp[2:-2,2:-2,:,M.tau-1] + self.t0[2:-2,2:-2,:]
       b=M.maskt[2:-2,2:-2,:]
       a=sum(a*b,axis=0)/sum(b,axis=0)
       co=ax.contourf(y,z,a.transpose())
       self.figure.colorbar(co)
       ax.set_title('zonal average');
       ax.axis('tight')
     return

 

if __name__ == "__main__": channel1().mainloop( snapint = 86400.0)


