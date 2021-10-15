
import numpy

class pyOMError(Exception):
   """ generic exception of class pyOM
   """ 
   def __init__(self, value):
       self.value = value
   def __str__(self):
       return repr(self.value)

class pyOM:
   """
   Main class describing the model
   """
   
   def __init__(self):
     """ initialize everyting
     """
     # try to load module and fortran code with MPI bindings or try to live without MPI
     try:     
        from mpi4py import MPI
        import pyOM_code_MPI 
        self.fortran = pyOM_code_MPI
        print ' checking MPI'
        self.mpi_comm = MPI.COMM_WORLD  
        self.fortran.my_mpi_init( self.mpi_comm.py2f())
        # determine domain decomposition from commandline input
        M=self.fortran.main_module        # fortran module with model variables
        if M.n_pes>1:
          import sys
          if len(sys.argv) < 3: raise pyOMError(' not enough parameter')
          M.n_pes_i = sys.argv[1]
          M.n_pes_j = sys.argv[2]
          if M.my_pe==0: print 'using ',M.n_pes_i,' x ',M.n_pes_j ,' PEs'
     except ImportError: 
        import pyOM_code
        self.fortran = pyOM_code
        self.fortran.my_mpi_init(0) 
     self.pyOM_version = self.fortran.main_module.version
     
     # set all model parameter by user-defined overloaded method
     self.set_parameter()        
     
     # set domain decomposition for parallel execution
     self.fortran.pe_decomposition() 
     
     # allocate model variables in fortran modules
     self.fortran.allocate_main_module() 
     self.fortran.allocate_isoneutral_module()
     self.fortran.allocate_tke_module()
     self.fortran.allocate_eke_module()
     self.fortran.allocate_idemix_module()

     # setup the grid by user defined method
     self.set_grid()
     self.fortran.calc_grid()

     # set grid dependent stuff as coriolis parameter by user defined method
     self.set_coriolis()
     self.fortran.calc_beta()

     # set topography by user defined method
     self.set_topography()
     self.fortran.calc_topo()
     self.fortran.calc_spectral_topo()
     
     # set initial conditions by user defined method
     self.set_initial_conditions()   
     self.fortran.calc_initial_conditions()
     
     # set first time all periodic forcing by user defined method
     self.set_forcing()

     # setup island line integrals for streamfunction
     if self.fortran.main_module.enable_streamfunction: self.fortran.streamfunction_init()

     #! check setup
     if self.fortran.tke_module.enable_tke and not self.fortran.main_module.enable_implicit_vert_friction:
       if  self.fortran.main_module.my_pe==0: 
              print 'ERROR: use TKE model only with implicit vertical friction '
              print '        -> switch on enable_implicit_vert_fricton        '
       raise pyOMError

     self.fortran.check_isoneutral_slope_crit()
     return


   def run(self,snapint = -1 ,runlen = -1):
     """
     enter a simple model time stepping loop
     """
     
     M=self.fortran.main_module        

     if snapint <0: snapint = 10*M.dt_tracer
     if runlen  <0: runlen  = 100*M.dt_tracer

     snapint_itt = int(snapint/M.dt_tracer)
     enditt = M.itt+int(runlen/M.dt_tracer)
     startitt = M.itt*1
     if M.my_pe == 0 : 
        print 'Starting integration for ',runlen,' s/ ',int(runlen/M.dt_tracer),' time steps'
        print ' from time step ',startitt,' to ',enditt-1
        print ' with snapshot interval of ',snapint_itt,' time steps'

     for M.itt in range(startitt,enditt):
       self.time = M.itt*M.dt_tracer
       self.time_step()
       if numpy.mod(M.itt,snapint_itt) < 1 :  self.diagnose()
       self.time_goes_by()
     M.itt = M.itt+1  
     
     if M.my_pe==0: print ' end of integration '  
     return
   

   def time_step(self):
     """ do one time step
     """
     M=self.fortran.main_module        
     I=self.fortran.idemix_module      
     E=self.fortran.eke_module        
     T=self.fortran.tke_module       
     self.set_forcing()
       
     if I.enable_idemix:                           self.fortran.set_idemix_parameter()
     if I.enable_idemix_m2 or I.enable_idemix_niw: self.fortran.set_spectral_parameter()
           
     self.fortran.set_eke_diffusivities()
     self.fortran.set_tke_diffusivities()
        
     self.fortran.momentum()
     self.fortran.thermodynamics()
       
     if E.enable_eke or T.enable_tke or I.enable_idemix: self.fortran.calculate_velocity_on_wgrid()

     if E.enable_eke: self.fortran.integrate_eke()

     if I.enable_idemix_m2:   self.fortran.integrate_idemix_m2()
     if I.enable_idemix_niw:  self.fortran.integrate_idemix_niw()
     if I.enable_idemix:      self.fortran.integrate_idemix()
     if I.enable_idemix_m2 or I.enable_idemix_niw:  self.fortran.wave_interaction()

     if T.enable_tke:    self.fortran.integrate_tke()

     #---------------------------------------------------------------------------------
     # Main boundary exchange
     # for density, temp and salt this is done in integrate_tempsalt.f90
     #---------------------------------------------------------------------------------
     self.fortran.border_exchg_xyz(M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,M.u[:,:,:,M.taup1-1],M.nz) 
     self.fortran.setcyclic_xyz   (M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,M.u[:,:,:,M.taup1-1],M.nz)
     self.fortran.border_exchg_xyz(M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,M.v[:,:,:,M.taup1-1],M.nz) 
     self.fortran.setcyclic_xyz   (M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,M.v[:,:,:,M.taup1-1],M.nz)
     
     if T.enable_tke: 
         self.fortran.border_exchg_xyz(M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,T.tke[:,:,:,M.taup1-1],M.nz) 
         self.fortran.setcyclic_xyz   (M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,T.tke[:,:,:,M.taup1-1],M.nz)
     if E.enable_eke: 
         self.fortran.border_exchg_xyz(M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,E.eke[:,:,:,M.taup1-1],M.nz) 
         self.fortran.setcyclic_xyz   (M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,E.eke[:,:,:,M.taup1-1],M.nz)
     if I.enable_idemix: 
         self.fortran.border_exchg_xyz(M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,I.e_iw[:,:,:,M.taup1-1],M.nz) 
         self.fortran.setcyclic_xyz   (M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,I.e_iw[:,:,:,M.taup1-1],M.nz)
     if I.enable_idemix_m2: 
         self.fortran.border_exchg_xyp(M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,I.e_m2[:,:,:,M.taup1-1],I.np) 
         self.fortran.setcyclic_xyp   (M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,I.e_m2[:,:,:,M.taup1-1],I.np)
     if I.enable_idemix_niw: 
         self.fortran.border_exchg_xyp(M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,I.e_niw[:,:,:,M.taup1-1],I.np) 
         self.fortran.setcyclic_xyp   (M.is_pe-M.onx,M.ie_pe+M.onx,M.js_pe-M.onx,M.je_pe+M.onx,I.e_niw[:,:,:,M.taup1-1],I.np)
       
     if M.enable_hydrostatic: self.fortran.vertical_velocity()
     return

   def time_goes_by(self):
       """ shift time levels by one
       """
       M=self.fortran.main_module        
       store   = M.taum1*1 # to create new instance !!!
       M.taum1 = M.tau
       M.tau   = M.taup1
       M.taup1 = store
       return

   def read_restart(self):
       """ read restart file
       """
       self.fortran.main_module.itt = self.fortran.read_restart()
       return
   
   def write_restart(self):
       """ write restart file
       """
       self.fortran.write_restart(self.fortran.main_module.itt)
       return
   
   def diagnose(self):
       """ diagnose the model variables, might be extended by further diagnostic
       """
       M=self.fortran.main_module        
       if M.my_pe==0:
         print 'diagnosing at %f s, itt=%i, solver itt = %i' %(M.itt*M.dt_tracer,M.itt,M.congr_itts)
       return

   def if2py(self,i):
       """ converts fortran zonal index i to python indexing
       """
       return i+self.fortran.main_module.onx-self.fortran.main_module.is_pe
   
   def jf2py(self,j):
       """ converts fortran meridional index j to python indexing
       """
       return j+self.fortran.main_module.onx-self.fortran.main_module.js_pe

   def ip2fy(self,i):
       """ converts python zonal index i to fortran indexing
       """
       return  i+self.fortran.main_module.is_pe-self.fortran.main_module.onx
   
   def jp2fy(self,j):
       """ converts python meridional index j to fortran indexing
       """
       return  j+self.fortran.main_module.js_pe-self.fortran.main_module.onx
   
   
   def set_parameter(self):
       """set main parameter 
          this is a dummy routines which needs to be overloaded
       """
       return

   def set_grid(self):
       """ set grid, i.e. dxt, dyt, and dzt 
           this is a dummy routines which needs to be overloaded
       """
       return
   
   def set_coriolis(self):
       """  set coriolis parameter coriolis_t
            this is a dummy routines which needs to be overloaded
       """
       return

   def set_topography(self):
       """  set topography kbot
            this is a dummy routines which needs to be overloaded
       """
       self.fortran.main_module.kbot[:]=1
       return

   def set_initial_conditions(self):
       """  set initial conditions
            this is a dummy routines which needs to be overloaded
       """
       return
       
   def set_forcing(self):
       """  set periodic forcing
            this is a dummy routines which needs to be overloaded
       """
       return
        
if __name__ == "__main__": print 'I will do nothing'
