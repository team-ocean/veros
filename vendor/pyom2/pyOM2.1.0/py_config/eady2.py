
import sys; sys.path.append('../py_src')
from pyOM_gui import pyOM_gui as pyOM
from numpy import *
import lin_stab

#H0  = 200   # water depth
#N0  = 0.001 # stability freq.
#EPS = 1e-6
#U0  = 0.15  # velocity mag.


H0  = 2000   # water depth
N0  = 0.004  # stability freq.
U0  = 0.5    # velocity mag.
EPS = 1e-5

f0  = 1e-4   # coriolis freq.
beta= 0e-11  # beta 
HY  = 0.0e-3 # slope of topography

KX_INITAL = 1.0  # initial with wave with wavenumber k = KX_INITAL * kmax
                 # where kmax is wavenumber of fastest growing mode

class eady2(pyOM):
   """ Eady (1941) solution
   """
   def set_parameter(self):
     """set main parameter
     """
     M=self.fortran.main_module   
     (M.nx,M.ny,M.nz)    = (34,34,10)

     M.congr_epsilon = 1e-12
     M.congr_max_iterations = 5000
     M.enable_streamfunction = 1
     
     M.enable_superbee_advection  = 1
     M.enable_explicit_vert_friction = 1
     M.enable_hor_friction = 1
     M.kappam_0   = 1.e-4 
     M.kappah_0   = 1.e-4 
     
     M.enable_hydrostatic      = 1
     M.enable_cyclic_x         = 1
     M.enable_conserve_energy  = 0
     M.coord_degree            = 0
     M.eq_of_state_type        = 1 
     M.enable_tempsalt_sources = 1
     return
   
   def set_grid(self):
     M=self.fortran.main_module   
     # print some numbers
     Lr    = N0*H0/f0          # Rossby radius
     delta = H0/Lr             # aspect ratio
     Ro    = U0/(f0*Lr)       # Rossby number
     Ri    = N0**2*H0**2/U0**2 # Richardson number
     print
     print  ' L  = %f km'%(Lr/1e3)
     print  ' Ro = %f '%Ro
     print  ' Ri = %f '%Ri
     print  ' delta = %f '%delta
     print  ' ell = %f '%(Lr/6400e3)
     print
     # solve linear stability problem first
     ksx=linspace(0,3.2,40);    kx=ksx/Lr
     ky = array( [0./Lr] )
     M.dzt[:]    = H0/M.nz
     print  ' Delta z = ',M.dzt[0]
     zw=arange(M.nz)*M.dzt[0]+ M.dzt[0]
     zt=zw-M.dzt[0]/2.0
     U=U0/2+U0*zt/H0
     V=U*0
     B=N0**2*zt
     if 1:
      om_max,om,kmax,lmax,u,v,w,b,p=lin_stab.qg(U,V,B,M.dzt[0],kx,ky,beta,f0,0,HY)
      print ' Max. growth rate QG %f 1/days ' % (-imag(om)*86400)
      print ' k_max Lr = %f  , l_max L_r = %f ' % (kmax*Lr/pi,lmax*Lr/pi)
     if 0:
      om_max,om,kmax,lmax,u,v,w,b,p=lin_stab.pe(U,V,B,M.dzt[0],kx,ky,0.,beta,0.,f0,0.,HY)
      print ' Max. growth rate PE %f 1/days ' % (-imag(om)*86400)
      print ' k_max = %f Lr , l_max = %f Lr' % (kmax*Lr,lmax*Lr)
     print
     self.lin_stab_kmax = kmax
     self.lin_stab_b = b
     self.lin_stab_p = p
     self.lin_stab_u = u
     self.lin_stab_v = v
     self.lin_stab_w = w
     L = 2*2*pi/kmax    # two waves
     M.dxt[:]  =   L/M.nx 
     M.dyt[:]  =   L/M.nx 
     M.dt_tracer  = M.dxt[0]*1200/20e3   # c = 20e3/1200.0,  dt =dx/c
     M.dt_mom     = M.dxt[0]*1200/20e3   # c = 20e3/1200.0,  dt =dx/c
     print "dx=%f km, dt= %f s "%(M.dxt[0]/1e3,M.dt_tracer)
     #M.a_h = (M.dxt[0])**3*2e-11    
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
     grav = 9.81; rho0 = 1024.0
 
     # background velocity
     for k in range(M.nz):
        M.u[:,:,k,M.tau-1]=(U0/2+U0*M.zt[k]/H0)*M.masku[:,:,k]

     # background buoyancy
     self.t0  = zeros( (M.i_blk+2*M.onx,M.j_blk+2*M.onx,M.nz) , 'd', order='F')
     self.dt0 = zeros( (M.i_blk+2*M.onx,M.j_blk+2*M.onx,M.nz,3) , 'd', order='F')
     
     for k in range(M.nz):
        self.t0[:,:,k]=-N0**2*M.zt[k]/grav/alpha*rho0*M.maskt[:,:,k]
        
     for k in range(M.nz):
      for j in range(1,M.ny+1):
        jj = self.jf2py(j)
        f   = (M.coriolis_t[0,jj]+M.coriolis_t[0,jj+1])/2.0
        uz = U0/M.ht[:,jj]
        self.t0[:,jj+1,k]=(self.t0[:,jj,k]+M.dyt[jj]*uz*f/grav/alpha*rho0)*M.maskt[:,jj,k]

     # perturbation buoyancy, etc
     for k in range(M.nz):
      for j in range(M.yt.shape[0]):
        ky=pi/(M.ny*M.dyt[0])
        kx = KX_INITAL*self.lin_stab_kmax
        phase = cos(kx*M.xt) +complex(0,1)*sin(kx*M.xt)
        phase = phase*sin(ky*M.yt[j])
        amp = 0.2
        M.temp[:,j,k,M.tau-1] = amp*real(phase*self.lin_stab_b[k])*M.maskt[:,j,k]*rho0/(grav*alpha)
        M.u[:,j,k,M.tau-1]    = M.u[:,j,k,M.tau-1] + amp*real(phase*self.lin_stab_u[k])*M.masku[:,j,k]
        M.v[:,j,k,M.tau-1] =                         amp*real(phase*self.lin_stab_v[k])*M.maskv[:,j,k]
        M.w[:,j,k,M.tau-1] =                         amp*real(phase*self.lin_stab_w[k])*M.maskw[:,j,k]
        
     for d in (M.temp,M.u,M.v,M.w): d[...,M.taum1-1] = d[...,M.tau-1]

     return


   def topography(self):
     """ setup topography
     """
     M=self.fortran.main_module
     #for j in range(M.ny): 
     #   z0 = M.yt[j]*HY
     #   for k in range(M.nz):
     #     if z0 > M.zt[k]-M.zt[0] :
     #       M.maskt[:,j,k]=0
     M.kbot[:]=0
     M.kbot[:,2:-2]=1
     return



   def make_plot(self):
     """ make a plot using methods of self.figure
     """
     if hasattr(self,'figure'):
       M=self.fortran.main_module         # fortran module with model variables
       k=M.nz*3/4
       k2=M.nz/4
       i=int(M.nx/2)
       j=int(M.ny/2)
       x=M.xt[2:-2]/1e3
       y=M.yt[2:-2]/1e3
       z=M.zt[:]

       self.figure.clf()
       ax=self.figure.add_subplot(221)
       a=M.temp[2:-2,2:-2,:,M.tau-1] + self.t0[2:-2,2:-2,:]
       b=M.maskt[2:-2,2:-2,:]
       a=sum(a*b,axis=0)/sum(b,axis=0)
       co=ax.contourf(y,z,a.transpose())
       a=M.u[2:-2,2:-2,:,M.tau-1] 
       b=M.masku[2:-2,2:-2,:]
       a=sum(a*b,axis=0)/sum(b,axis=0)
       ax.contour(y,z,a.transpose(),10,colors='black')
       ax.set_ylabel('z [km]')

       self.figure.colorbar(co)
       ax.set_title("total temp. at x=%4.1f km"%x[i]);  
       ax.axis('tight') 
         
       ax=self.figure.add_subplot(222)
       co=ax.contourf(x,y,M.temp[2:-2,2:-2,k,M.tau-1].transpose())
       a=M.temp[2:-2,2:-2,k,M.tau-1] + self.t0[2:-2,2:-2,k]
       ax.contour(x,y,a.transpose(),10,colors='black')
       ax.set_title("temp. perturb. at z=%4.1f m"%z[k]); 
       self.figure.colorbar(co)
       ax.axis('tight')
       
       ax=self.figure.add_subplot(223)
       co=ax.contourf(x,y,M.temp[2:-2,2:-2,k2,M.tau-1].transpose())
       a=M.temp[2:-2,2:-2,k2,M.tau-1] + self.t0[2:-2,2:-2,k2]
       ax.contour(x,y,a.transpose(),10,colors='black')
       ax.set_title("temp. perturb. at z=%4.1f m"%z[k2]); 
       a=M.u[2:-2:2,2:-2:2,k,M.tau-1] 
       b=M.v[2:-2:2,2:-2:2,k,M.tau-1] 
       ax.quiver(x[::2],y[::2],a.transpose(),b.transpose() )
       self.figure.colorbar(co)
       ax.axis('tight')
          
       
       ax=self.figure.add_subplot(224)
       a=M.temp[2:-2,j,:,M.tau-1] 
       co=ax.contourf(x,z,a.transpose())
       ax.set_title("temp. perturb. at y=%4.1f km"%y[j]); 
       ax.set_xlabel('x [km]'); 
       self.figure.colorbar(co)
       ax.axis('tight')
     return

if __name__ == "__main__": eady2().mainloop( snapint = 86400.0)


