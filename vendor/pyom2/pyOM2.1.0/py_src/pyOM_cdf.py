#!/usr/bin/python

import numpy
from pyOM import pyOM

class pyOM_cdf(pyOM):
   """
   pyOM with snapshot output in netcdf format
   """
   def __init__(self):
     pyOM.__init__(self)
     try:     # try to load module with netcdf bindings
              from netCDF4 import Dataset as NF
              self.NF=NF
     except ImportError:
              from Scientific.IO.NetCDF import NetCDFFile as NF
              self.NF=NF
     # note that scipy.io.netcdf does not support appending data to file
     # therefore we cannot use that module
     self.snap_file = 'pyOM.cdf'
     self._spval = -1e33*numpy.ones( (1,),'f')  
     self.init_snap_cdf()
     return
  
   def diagnose(self):
     """ diagnose the model variables
     """
     pyOM.diagnose(self)
     self.write_snap_cdf()
     return

   def init_snap_cdf(self):
     """ intitialize netcdf diagnostics
     """
     M=self.fortran.main_module        
     self.define_grid_cdf(self.snap_file)
     if M.my_pe == 0:
       fid = self.NF(self.snap_file,'a')
   
       id=fid.createVariable('temp','f',('Time','zt','yt','xt') )
       id.long_name = 'Temperature'; id.units='deg C'
       id.missing_value = self._spval
  
       id=fid.createVariable('salt','f',('Time','zt','yt','xt') )
       id.long_name = 'Salinity'; id.units='g/kg'
       id.missing_value = self._spval
  
       id=fid.createVariable('u','f',('Time','zt','yt','xu') )
       id.long_name = 'Zonal velocity'; id.units='m/s'
       id.missing_value = self._spval
  
       id=fid.createVariable('v','f',('Time','zt','yu','xt') )
       id.long_name = 'Meridional velocity'; id.units='m/s'
       id.missing_value = self._spval
  
       id=fid.createVariable('w','f',('Time','zw','yt','xt') )
       id.long_name = 'Vertical velocity'; id.units='m/s'
       id.missing_value = self._spval
       
       id=fid.createVariable('taux','f',('Time','yt','xu') )
       id.long_name = 'Zonal wind stress'; id.units='m^2/s^2'
       id.missing_value = self._spval
  
       id=fid.createVariable('tauy','f',('Time','yt','xu') )
       id.long_name = 'Meridional wind stress'; id.units='m^2/s^2'
       id.missing_value = self._spval
       
       id=fid.createVariable('forc_temp_surface','f',('Time','yt','xt') )
       id.long_name = 'Surface temperature flux'; id.units='deg C/m^2/s'
       id.missing_value = self._spval
       
       id=fid.createVariable('forc_salt_surface','f',('Time','yt','xt') )
       id.long_name = 'Surface salinity flux'; id.units='1/m^2/s'
       id.missing_value = self._spval
       
       id=fid.createVariable('psi','f',('Time','yu','xu') )
       id.long_name = 'streamfunction'; id.units='m^3/s'
       id.missing_value = self._spval
       
       I=self.fortran.idemix_module        
       if I.enable_idemix:
           id=fid.createVariable('forc_iw_bottom','f',('Time','yt','xt') )
           id.long_name = 'Internal wave forcing'; id.units='W/m^2'
           id.missing_value = self._spval
           
           id=fid.createVariable('forc_iw_surface','f',('Time','yt','xt') )
           id.long_name = 'Internal wave forcing'; id.units='W/m^2'
           id.missing_value = self._spval
           
       fid.close()
     return
   
   def mask_me(self,a,m):
         """ apply mask m to variable a
         """
         M=self.fortran.main_module         # fortran module with model variables
         a = a[ self.if2py(M.is_pe):self.if2py(M.ie_pe)+1 , self.jf2py(M.js_pe):self.jf2py(M.je_pe)+1]
         m = m[ self.if2py(M.is_pe):self.if2py(M.ie_pe)+1 , self.jf2py(M.js_pe):self.jf2py(M.je_pe)+1]
         return numpy.where( m ==0, a*0+self._spval, a)

   def write_snap_cdf(self):
     """  write netcdf snapshot, append file to record dimension Time
     """
     M=self.fortran.main_module         # fortran module with model variables
     if M.my_pe == 0:   
          fid= self.NF(self.snap_file,'a')
          tid=fid.variables['Time'];
          i=list(numpy.shape(tid))[0];
          tid[i]=M.itt*M.dt_tracer

     a = numpy.zeros( (M.nx,M.ny) , 'd', order='F') 
     NaN = a+self._spval
       
     a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me( M.psi[ :,:,M.tau-1], M.maskz[:,:,-1] )
     self.fortran.pe0_recv_2d(a)
     if M.my_pe == 0: fid.variables['psi'][i,:] = a.transpose().astype('f')
           
     a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me( M.surface_taux[:], M.masku[:,:,-1] )
     self.fortran.pe0_recv_2d(a)
     if M.my_pe == 0: fid.variables['taux'][i,:] = a.transpose().astype('f')
           
     a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me( M.surface_tauy[:], M.maskv[:,:,-1] )
     self.fortran.pe0_recv_2d(a)
     if M.my_pe == 0: fid.variables['tauy'][i,:] = a.transpose().astype('f')
           
     a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me( M.forc_temp_surface[:], M.maskt[:,:,-1] )
     self.fortran.pe0_recv_2d(a)
     if M.my_pe == 0: fid.variables['forc_temp_surface'][i,:] = a.transpose().astype('f')
           
     a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me( M.forc_salt_surface[:], M.maskt[:,:,-1] )
     self.fortran.pe0_recv_2d(a)
     if M.my_pe == 0: fid.variables['forc_salt_surface'][i,:] = a.transpose().astype('f')
           
     I=self.fortran.idemix_module        
     if I.enable_idemix:
       a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me( I.forc_iw_bottom[:], M.maskw[:,:,-1] )
       self.fortran.pe0_recv_2d(a)
       if M.my_pe == 0: fid.variables['forc_iw_bottom'][i,:] = a.transpose().astype('f')
           
       a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me( I.forc_iw_surface[:], M.maskw[:,:,-1] )
       self.fortran.pe0_recv_2d(a)
       if M.my_pe == 0: fid.variables['forc_iw_surface'][i,:] = a.transpose().astype('f')
           

         
     for k in range(M.nz):
          a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me(M.temp[:,:,k,M.tau-1],M.maskt[:,:,k])
          self.fortran.pe0_recv_2d(a)
          if M.my_pe == 0: fid.variables['temp'][i,k,:] = a.transpose().astype('f')
              
          a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me(M.salt[:,:,k,M.tau-1],M.maskt[:,:,k])
          self.fortran.pe0_recv_2d(a)
          if M.my_pe == 0: fid.variables['salt'][i,k,:] = a.transpose().astype('f')
              
          a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me(M.u[:,:,k,M.tau-1],M.masku[:,:,k])
          self.fortran.pe0_recv_2d(a)
          if M.my_pe == 0: fid.variables['u'][i,k,:] = a.transpose().astype('f')
              
          a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me(M.v[:,:,k,M.tau-1],M.maskv[:,:,k])
          self.fortran.pe0_recv_2d(a)
          if M.my_pe == 0: fid.variables['v'][i,k,:] = a.transpose().astype('f')
              
          a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me(M.w[:,:,k,M.tau-1],M.maskw[:,:,k])
          self.fortran.pe0_recv_2d(a)
          if M.my_pe == 0: fid.variables['w'][i,k,:] = a.transpose().astype('f')
              
     if M.my_pe == 0 : fid.close()
     return   

   def define_grid_cdf(self,filename):
       """ define a netcdf file with dimensions as in model
       """
       M=self.fortran.main_module       
       if M.my_pe == 0:
         fid = self.NF(filename,'w')
         fid.createDimension('xt',int(M.nx))
         fid.createDimension('xu',int(M.nx))
         fid.createDimension('yt',int(M.ny))
         fid.createDimension('yu',int(M.ny))
         fid.createDimension('zt',int(M.nz))
         fid.createDimension('zw',int(M.nz))
         fid.createDimension('Time',None)
       
         Time=fid.createVariable('Time','f',('Time',) ) 
         Time.long_name = 'Time since start'; Time.units='Seconds'
         Time.time_origin='01-JAN-1900 00:00:00'

         xt=fid.createVariable('xt','f',('xt',) )
         xu=fid.createVariable('xu','f',('xu',) )
         yt=fid.createVariable('yt','f',('yt',) )
         yu=fid.createVariable('yu','f',('yu',) )
         
         if M.coord_degree:
           xt.long_name = 'Longitude on T grid'; xt.units = 'degrees E'
           xu.long_name = 'Longitude on U grid'; xu.units = 'degrees E'
           yt.long_name = 'Latitude on T grid';  yt.units = 'degrees N'
           yu.long_name = 'Latitude on U grid';  yu.units = 'degrees N'
         else:
            xt.long_name='Zonal coordinate on T grid'; xt.units='m'
            xu.long_name='Zonal coordinate on U grid'; xu.units='m'
            yt.long_name='Meridional coordinate on T grid'; yt.units='m'
            yu.long_name='Meridional coordinate on V grid'; yu.units='m'

         zt=fid.createVariable('zt','f',('zt',) )
         zt.long_name='Vertical coordinate on T grid'; zt.units='m'
         zw=fid.createVariable('zw','f',('zw',) )
         zw.long_name='Vertical coordinate on W grid'; zw.units='m'
       
         zt[:]=M.zt.astype('f')
         zw[:]=M.zw.astype('f')
         
       a = numpy.zeros( (M.nx,M.ny) , 'd', order = 'F')
       
       a[M.is_pe-1:M.ie_pe,0] = M.xt[2:-2]
       self.fortran.pe0_recv_2d(a)
       if M.my_pe == 0:   xt[:]=a[:,0].astype('f')
           
       a[M.is_pe-1:M.ie_pe,0] = M.xu[2:-2]
       self.fortran.pe0_recv_2d(a)
       if M.my_pe == 0:   xu[:]=a[:,0].astype('f')

       a[0,M.js_pe-1:M.je_pe] = M.yt[2:-2]
       self.fortran.pe0_recv_2d(a)
       if M.my_pe == 0:  yt[:]=a[0,:].astype('f')
           
       a[0,M.js_pe-1:M.je_pe] = M.yu[2:-2]
       self.fortran.pe0_recv_2d(a)
       if M.my_pe == 0: yu[:]=a[0,:].astype('f')
       return

   

if __name__ == "__main__":
   print 'I will do nothing'
