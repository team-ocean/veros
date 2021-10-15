
import sys; sys.path.append('../py_src')
from pyOM_gui import pyOM_gui as pyOM
from numpy import *
import Tkinter



class eq_waves1(pyOM):
   """ Equatorial waves
   """
   def set_parameter(self):
     """set main parameter
     """
     HRESOLVE=1.0
     M=self.fortran.main_module   
     M.nx    = 64
     M.nz    = 1
     M.ny    = 64
     M.dt_tracer    = 3600.0/2.
     M.dt_mom       = 3600.0/2.
     
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
     M.enable_biharmonic_friction  = 0
     M.a_hbi  = 5e11
     return

   def set_grid(self):
     M=self.fortran.main_module   
     M.dxt[:] = 100e3
     M.dyt[:] = 100e3
     M.dzt[:] = (Re.get()**2*BETA.get()) **2 /M.grav
     return
 
   def set_coriolis(self):
     """ vertical and horizontal Coriolis parameter on yt grid
         routine is called after initialization of grid
     """
     M=self.fortran.main_module   
     y0=M.ny*M.dxt[0]*0.5
     for j in range( M.yt.shape[0] ): M.coriolis_t[:,j]   =  BETA.get()*(M.yt[j]-y0)
     return

   def set_initial_conditions(self):
     """ setup all initial conditions
     """
     M=self.fortran.main_module   
     cn =  (M.dzt[0]*9.81)**0.5  
     hn=cn**2/9.81
     y0=M.ny*M.dxt[0]*0.5
     A=0.01
     print
     print 'h_n = ',M.dzt[0],' m'
     print 'R_e = ',Re.get()/1e3,' km'
     print "T_e = 1/sqrt(beta c) = " , 1/(BETA.get()*cn)**0.5 /86400. , ' days'
     
     if Kelvin.get():# Kelvin wave
       kx=KX.get()*pi/( M.nx*M.dxt[0] )
       omega =cn*kx
       for i in range(M.xt.shape[0]):
        for j in range(M.yt.shape[0]):
          y=M.yt[j]-y0
          M.v[i,j,0,:]= 0.0
          M.u[i,j,0,:]= A*exp( -0.5*y**2/Re.get()**2 )*cos(kx*M.xu[i]) /cn
          M.psi[i,j,:]= A*exp( -0.5*y**2/Re.get()**2 )*cos(kx*M.xt[i]) 
        
     if Yanai.get():# Yanai wave  #  beta c_n + (k c_n/2)^2  =  omega^2 -k c_n omega + (k c_n/2)^2
                        #                          = (omega - k c_n/2)^2
                        #  omega = k c_n/2 +  (beta c_n + (k c_n/2)^2 )^0.5
       kx=KX.get()*pi/( M.nx*M.dxt[0] )
       omega = kx*cn/2 + (BETA.get()*cn+(kx*cn/2)**2 )**0.5
       for i in range(M.xt.shape[0]):
        for j in range(M.yt.shape[0]):
          y=M.yt[j]-y0
          yu=M.yu[j]-y0
          M.v[i,j,0,:]=A*exp( -0.5*yu**2/Re.get()**2 )*cos(kx*M.xt[i])
          M.u[i,j,0,:]=-A*BETA.get()*y/(omega-cn*kx)*exp( -0.5*y**2/Re.get()**2 )*sin(kx*M.xu[i])
          M.psi[i,j,:]=-A*BETA.get()*y/(omega-cn*kx)*exp( -0.5*y**2/Re.get()**2 )*sin(kx*M.xt[i])*cn

     if Gravity.get():# Gravity wave  omega^2 = cn^2(kx^2+2m/Re^2+1/Re^2)
       m=mMode.get()
       kx=KX.get()*pi/( M.nx*M.dxt[0] )
       omega= cn*(kx**2+2*m/Re.get()**2+1/Re.get()**2)**0.5
       for i in range(M.xt.shape[0]):
        for j in range(M.yt.shape[0]):
          y=M.yt[j]-y0
          yu=M.yu[j]-y0
          M.v[i,j,0,M.tau-1]   = A*Hf(m,yu/Re.get())*cos(kx*M.xt[i])
          M.v[i,j,0,M.taum1-1] = A*Hf(m,yu/Re.get())*cos(kx*M.xt[i]+omega*M.dt_tracer)
          pp           = Hf(m+1,y/Re.get())/(omega-cn*kx)+2*m*Hf(m-1,y/Re.get())/(omega+cn*kx) 
          M.u[i,j,0,M.tau-1]   = -A*0.5* (cn*BETA.get())**0.5*( pp )*sin(kx*M.xu[i])
          M.u[i,j,0,M.taum1-1] = -A*0.5* (cn*BETA.get())**0.5*( pp )*sin(kx*M.xu[i]+omega*M.dt_tracer)
          pp           = Hf(m+1,y/Re.get())/(omega-cn*kx)-2*m*Hf(m-1,y/Re.get())/(omega+cn*kx) 
          M.psi[i,j,M.tau-1]   = -A*cn*0.5* (cn*BETA.get())**0.5*( pp )*sin(kx*M.xt[i]) 
          M.psi[i,j,M.taum1-1] = -A*cn*0.5* (cn*BETA.get())**0.5*( pp )*sin(kx*M.xt[i]+omega*M.dt_tracer) 


     if Rossby.get():# Rossby wave  omega = - beta k /(k^2+2m/Re^2+1/Re^2)
       m=mMode.get()
       kx=KX.get()*pi/( M.nx*M.dxt[0] )
       omega= - BETA.get()*kx/(kx**2+2*m/Re.get()**2 + 1/Re.get()**2) 
       for i in range(M.xt.shape[0]):
        for j in range(M.yt.shape[0]):
          y=M.yt[j]-y0
          yu=M.yu[j]-y0
          M.v[i,j,0,M.tau-1  ] = A*Hf(m,yu/Re.get())*cos(kx*M.xt[i])
          M.v[i,j,0,M.taum1-1] = A*Hf(m,yu/Re.get())*cos(kx*M.xt[i]+omega*M.dt_tracer)
          pp           = Hf(m+1,y/Re.get())/(omega-cn*kx)+2*m*Hf(m-1,y/Re.get())/(omega+cn*kx) 
          M.u[i,j,0,M.tau-1  ] = -A*0.5* (cn*BETA.get())**0.5*( pp )*sin(kx*M.xu[i])
          M.u[i,j,0,M.taum1-1] = -A*0.5* (cn*BETA.get())**0.5*( pp )*sin(kx*M.xu[i]+omega*M.dt_tracer)
          pp           = Hf(m+1,y/Re.get())/(omega-cn*kx)-2*m*Hf(m-1,y/Re.get())/(omega+cn*kx) 
          M.psi[i,j,M.tau-1  ] = -A*cn*0.5* (cn*BETA.get())**0.5*( pp )*sin(kx*M.xt[i]) 
          M.psi[i,j,M.taum1-1] = -A*cn*0.5* (cn*BETA.get())**0.5*( pp )*sin(kx*M.xt[i]+omega*M.dt_tracer) 
     return


   def config_setup(self):
      if hasattr(self,'config_frame'):
        Tkinter.Checkbutton(self.config_frame,text='Kelvin wave',variable=Kelvin).pack(side=Tkinter.TOP,anchor='w')
        Tkinter.Checkbutton(self.config_frame,text='Yanai wave',variable=Yanai).pack(side=Tkinter.TOP,anchor='w')
        Tkinter.Checkbutton(self.config_frame,text='Rossby wave',variable=Rossby).pack(side=Tkinter.TOP,anchor='w')
        Tkinter.Checkbutton(self.config_frame,text='Gravity wave',variable=Gravity).pack(side=Tkinter.TOP,anchor='w')
        F=Tkinter.Frame(self.config_frame); F.pack(side=Tkinter.TOP,anchor='w')
        Tkinter.Label(F,text='Rossby radius').pack(side=Tkinter.TOP,anchor='w')
        Tkinter.Entry(F,textvariable=Re).pack(side=Tkinter.TOP,anchor='w')
        F=Tkinter.Frame(self.config_frame); F.pack(side=Tkinter.TOP,anchor='w')
        Tkinter.Label(F,text='zonal wave number').pack(side=Tkinter.TOP,anchor='w')
        Tkinter.Entry(F,textvariable=KX).pack(side=Tkinter.TOP,anchor='w')
        F=Tkinter.Frame(self.config_frame); F.pack(side=Tkinter.TOP,anchor='w')
        Tkinter.Label(F,text='meridional mode').pack(side=Tkinter.TOP,anchor='w')
        Tkinter.Entry(F,textvariable=mMode).pack(side=Tkinter.TOP,anchor='w')
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
       ax.axis('tight') 
       
       a=M.u[2:-2:2,2:-2:2,0,M.tau-1] 
       b=M.v[2:-2:2,2:-2:2,0,M.tau-1] 
       ax.quiver(x[::2],y[::2],a.transpose(),b.transpose() )
       self.figure.colorbar(co)
         
     return


def Hf(m,y):
   return exp(-0.5*y**2)*Hermite_polynomial(m,y)

def Hermite_polynomial(m,y):
   if m<0:
      return 0
   elif m==0:
      return 1
   elif m==1:
      return 2*y
   elif m>1:
      return 2*y*Hermite_polynomial(m-1,y)-2*(m-1)*Hermite_polynomial(m-2,y)
   else:
      # never reached
      raise RuntimeError
   return
   
Tkinter.Tk()
BETA    = Tkinter.DoubleVar();   BETA.set(2e-11)
Re      = Tkinter.DoubleVar();   Re.set(500e3)
Kelvin  = Tkinter.BooleanVar(); Kelvin.set(True) 
Yanai   = Tkinter.BooleanVar(); Yanai.set(False)
Rossby  = Tkinter.BooleanVar(); Rossby.set(False)
Gravity = Tkinter.BooleanVar(); Gravity.set(False)
KX      = Tkinter.DoubleVar();   KX.set(2.0)
mMode   = Tkinter.IntVar();     mMode.set(1)

if __name__ == "__main__":
   model= eq_waves1()
   model.run(snapint=0.5*86400.0,runlen=365*86400.)
