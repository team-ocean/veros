#!/usr/bin/python

import numpy
from pyOM_cdf import pyOM_cdf 

class pyOM_ave(pyOM_cdf):
   """
   pyOM_cdf with averaging diagnostics in netcdf format
   """
   def __init__(self):
       pyOM_cdf.__init__(self)
       self.nitts = 0
       self.diag = []
       self.set_diagnostics()
       return
  
   def set_diagnostics(self):
       """ register variables to be averaged here. This is a dummy routine to be replaced
       """
       return
   
   def diagnose(self):
       """ diagnose the model variables
       """
       pyOM_cdf.diagnose(self)
       self.write_average_cdf()
       return

   def time_step(self):
       """ do one time step
       and calculate cummulative sum of all variables to be averaged 
       """
       pyOM_cdf.time_step(self)
       self.nitts = self.nitts +1
       for d in self.diag: 
           if d['ave'].shape == d['var'].shape:
                       d['ave'] = d['ave'] + d['var']
           else:
                       d['ave'] = d['ave'] + d['var'][...,self.fortran.main_module.tau-1]
       return

   def register_average(self, name = '', long_name = '', units = '', grid = '', var=None):
       """ register a variable to be averaged. grid is string of length (len(var.shape))
           indicating the grid location: 'T' for cell center, 'U' face of cell
       """
       #check if name is in use
       for d in self.diag:
            if d['name'] == name: 
                print ' name already in use'
                return
       # append diagnostics
       self.diag.append({'name':name,'long_name':long_name,'units':units,'grid':grid} )
       self.diag[-1]['var']=var
       if var.shape[-1] == 3:
          self.diag[-1]['ave'] = numpy.zeros( var.shape[:-1], 'd', order='F')
       else:
          self.diag[-1]['ave'] = numpy.zeros( var.shape, 'd', order='F')
       return

   def write_average_cdf(self):
       """ write averaged variables to netCDF file
       """
       M=self.fortran.main_module       
       ave_file = ('average_%12i.cdf'%M.itt).replace(' ','0')
       
       self.define_grid_cdf(ave_file)
       if (M.my_pe == 0):
         #print "writing to file ",ave_file
         fid = self.NF(ave_file,'a')
         for d in self.diag:
           dims = []
           if   d['grid'][0] == 'T': dims.append('xt')
           elif d['grid'][0] == 'U': dims.append('xu')
           else: raise pyOMError('unkown grid in averaged variable' + d['name'])
              
           if   d['grid'][1] == 'T': dims.append('yt')
           elif d['grid'][1] == 'U': dims.append('yu')
           else: raise pyOMError('unkown grid in averaged variable' + d['name'])

           if len(d['grid']) >2:
             if   d['grid'][2] == 'T': dims.append('zt')
             elif d['grid'][2] == 'U': dims.append('zw')
             else: raise pyOMError('unkown grid in averaged variable' + d['name'])
           dims.append('Time')
           id=fid.createVariable(d['name'],'f',tuple(dims[::-1]) )
           id.long_name = d['long_name']; id.units=d['units']
           id.missing_value = self._spval
          
       if (M.my_pe == 0):
         tid=fid.variables['Time'];
         i=list(numpy.shape(tid))[0];
         tid[i]=M.itt*M.dt_tracer
         
       a = numpy.zeros( (M.nx,M.ny) , 'd', order='F') 
       NaN = a+self._spval
       
       for d in self.diag:
           
         if len(d['grid']) >2:
           if   d['grid'] == 'TTT' : mask = M.maskt
           elif d['grid'] == 'UTT' : mask = M.masku
           elif d['grid'] == 'TUT' : mask = M.maskv
           elif d['grid'] == 'TTU' : mask = M.maskw
           elif d['grid'] == 'UUT' : mask = M.maskz
           else: raise pyOMError('unkown grid in averaged variable' + d['name'])
         else:
           if   d['grid'] == 'TT' : mask = M.maskt[:,:,-1]
           elif d['grid'] == 'UT' : mask = M.masku[:,:,-1]
           elif d['grid'] == 'TU' : mask = M.maskv[:,:,-1]
           elif d['grid'] == 'UU' : mask = M.maskz[:,:,-1]
           else: raise pyOMError('unkown grid in averaged variable' + d['name'])
             
         if len(d['ave'].shape)==2:
           a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me(d['ave'][:]/max(1,self.nitts),mask)
           self.fortran.pe0_recv_2d(a)
           if M.my_pe == 0: fid.variables[d['name']][i,:] = a.transpose().astype('f')
         elif len( d['ave'].shape )==3:
           for k in range(M.nz):
             a[ M.is_pe-1:M.ie_pe, M.js_pe-1:M.je_pe] = self.mask_me(d['ave'][:,:,k]/max(1,self.nitts),mask[:,:,k])
             self.fortran.pe0_recv_2d(a)
             if M.my_pe == 0: fid.variables[d['name']][i,k,:] = a.transpose().astype('f')
         else: 
             raise pyOMError('unexpected shape for averaged variable' + d['name'])
           
       self.nitts = 0
       for d in self.diag: d['ave']=d['ave']*0

       if M.my_pe == 0: fid.close()
       return
   
   

if __name__ == "__main__":
   print 'I will do nothing'
