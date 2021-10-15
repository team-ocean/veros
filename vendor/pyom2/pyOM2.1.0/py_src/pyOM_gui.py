
import numpy,code,os
import Tkinter, tkMessageBox, tkFileDialog, ScrolledText
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from pyOM_ave import pyOM_ave


class pyOM_gui(Tkinter.Frame,pyOM_ave):
   """
   pyOM with graphical user interface
   """
   
   def __init__(self):
     """  initialize windows, etc
     """  
     pyOM_ave.__init__(self)
     M=self.fortran.main_module   
     self.halt     = 1
     self.semaphore = ['timestep','quit','snapshot','average','read_restart','write_restart','user_defined']
     
     if  M.my_pe == 0:
       self.pyOM_vstr = 'pyOM v%2.1f' % self.pyOM_version
       Tkinter.Frame.__init__(self,master=None)
       self.pack(expand=Tkinter.YES)
       self.master.title(self.pyOM_vstr)
       self.master.protocol('WM_DELETE_WINDOW', self.quit)

       # the menu
       menubar = Tkinter.Menu(self)
       menu = Tkinter.Menu(menubar, tearoff=0)
       menu.add_command(label="Read restart file", command=self.read)
       menu.add_command(label="Write restart file", command=self.write)
       menu.add_command(label="Save plot to file", command=self.printplot)
       menu.add_separator()
       menu.add_command(label="Exit", command=self.quit)
       menubar.add_cascade(label="File", menu=menu)
       menu = Tkinter.Menu(menubar, tearoff=0)
       menu.add_command(label="Start integration", command=self.start)
       menu.add_command(label="Stop integration", command=self.stop)
       menu.add_command(label="Reset integration", command=self.reset)
       menu.add_command(label="Plot", command=self.prep_plot)
       menubar.add_cascade(label="Control", menu=menu)
       menu = Tkinter.Menu(menubar, tearoff=0)
       menu.add_command(label="Shell", command=self.shell)
       menu.add_command(label="Config", command=self.config)
       menubar.add_cascade(label="Options", menu=menu)
       self.master.config(menu=menubar) 

       # plotting area on window
       #self.image_height=480; self.image_width=640
       self.image_height=500; self.image_width=800
       self.main_area= Tkinter.Canvas(self,height=self.image_height,width=self.image_width)
       self.main_area.pack(side=Tkinter.TOP)
       self._tkphoto = Tkinter.PhotoImage(master=self.main_area)
     
       # text area on window
       self.time_step_list = ScrolledText.ScrolledText(self,height=5,state=Tkinter.NORMAL,bd=2)
       self.time_step_list.pack(side=Tkinter.TOP,fill=Tkinter.X,expand=Tkinter.YES)

       # write some text
       self.time_step_list.insert(Tkinter.END,'Welcome to %s\n\n' % self.pyOM_vstr)
       self.time_step_list.insert(Tkinter.END,'grid size    : nx=%i ny=%i nz=%i \n' % (M.nx,M.ny,M.nz)  )
       self.time_step_list.insert(Tkinter.END,'time step    : %f s\n' % M.dt_tracer )
       self.time_step_list.yview(Tkinter.MOVETO,1)
       self.time_step_list.config(state=Tkinter.DISABLED)

       # Toolbar area on window
       Toolbar = Tkinter.Frame(self,relief=Tkinter.SUNKEN,bd=2)
       Toolbar.pack(side=Tkinter.TOP, expand=Tkinter.YES, fill=Tkinter.X)
     
       self.snapint = Tkinter.IntVar()#; self.snapint.set(0)
       self.plotint = Tkinter.IntVar()#; self.plotint.set(0)

       file = os.path.join(matplotlib.rcParams['datapath'], 'images', 'stock_right.ppm')
       self.image_button1 = Tkinter.PhotoImage(master=Toolbar, file=file)
       file = os.path.join(matplotlib.rcParams['datapath'], 'images', 'stock_close.ppm')
       self.image_button2 = Tkinter.PhotoImage(master=Toolbar, file=file)

       # start/stop button
       self.itt_str = Tkinter.StringVar()
       F = Tkinter.Frame(Toolbar); F.pack(side=Tkinter.LEFT)
       Tkinter.Label(F, text = self.pyOM_vstr,width=18).pack(side=Tkinter.TOP,anchor='w')
       Tkinter.Label(F, textvariable = self.itt_str,width=18).pack(side=Tkinter.TOP,anchor='w')
       Tkinter.Button(Toolbar, text = "Run" , image=self.image_button1,command=self.start).pack(side=Tkinter.LEFT)
       Tkinter.Button(Toolbar, text = "Stop", image=self.image_button2,command=self.stop).pack(side=Tkinter.LEFT)

       # scale bars for diag/plotting intervals
       self.aveint = Tkinter.IntVar()#; self.plotint.set(0)
       Tkinter.Scale(Toolbar,variable = self.aveint,label="averaging interval", length=120, \
             from_ = 1, to = 500, orient=Tkinter.HORIZONTAL).pack(side=Tkinter.LEFT)
       Tkinter.Scale(Toolbar,variable = self.snapint,label="snapshot interval", length=120, \
             from_ = 1, to = 500, orient=Tkinter.HORIZONTAL).pack(side=Tkinter.LEFT)
       B=Tkinter.Checkbutton(Toolbar,command=self.scale_both)
       B.select()
       B.pack(side=Tkinter.LEFT)
       self.scale_bar = Tkinter.Scale(Toolbar,variable = self.plotint,label="plotting interval", length=120, \
             from_ = 1, to = 500, orient=Tkinter.HORIZONTAL)
       self.scale_bar.pack(side=Tkinter.LEFT,padx=15)

       # buttons to disable plotting/snapshot
       F = Tkinter.Frame(Toolbar); F.pack(side=Tkinter.LEFT)
       self.enable_plotting = Tkinter.IntVar(); self.enable_plotting.set(1)
       if not hasattr(self,'make_plot'): self.enable_plotting.set(0)
       Tkinter.Checkbutton(F,text='enable plotting',variable=self.enable_plotting).pack(side=Tkinter.TOP,anchor='w')
       self.enable_snapshot = Tkinter.IntVar(); self.enable_snapshot.set(0)
       Tkinter.Checkbutton(F,text='enable snaphots',variable=self.enable_snapshot).pack(side=Tkinter.TOP,anchor='w')
       self.enable_average = Tkinter.IntVar(); self.enable_average.set(0)
       Tkinter.Checkbutton(F,text='enable averaging',variable=self.enable_average).pack(side=Tkinter.TOP,anchor='w')

       # Figure size
       self.figure=Figure()
       self.figure.set_figwidth(float(self.image_width)/self.figure.get_dpi() )
       self.figure.set_figheight(float(self.image_height)/self.figure.get_dpi() )
       self.figure.text(0.5,0.5,self.pyOM_vstr,ha='center',va='center',fontsize='xx-large',color='darkblue')

       # first plot
       self._plotit() 
       self.itt_str.set('time step %i' % M.itt)
       self.time = M.itt*M.dt_tracer
       self.prep_plot()
       self.shell_on=False
     self.run = self.mainloop
     return
   
   def set_signal(self,cmd):
       """ send a signal to other PEs, only called by PE0
       """
       i=numpy.zeros( (1,),'i', order='F')
       i[0]=self.semaphore.index(cmd)
       self.fortran.pe0_bcast_int(i)
       return

   def get_signal(self):
       """ get a signal from PE 0
       """
       i=numpy.zeros( (1,),'i', order='F')
       self.fortran.pe0_bcast_int(i)
       if i>len( self.semaphore) or i<0: raise pyOMError('wrong signal in get_signal')         
       return self.semaphore[i]
   
   def mainloop(self,snapint = None, runlen = None):
       """ enter main loop
       """
       M=self.fortran.main_module      
       if M.my_pe==0: 
           if snapint:
                self.snapint.set( int( snapint/self.fortran.main_module.dt_tracer) )
                self.aveint.set( int( snapint/self.fortran.main_module.dt_tracer) )
           self.scale_both()
           Tkinter.mainloop() # PE0 will only return when quit button is hit, it is now event driven
       else:
         stop = False
         while not stop:         # other PEs have to listen to signals from event driven PE0
          cmd = self.get_signal()         
          if   cmd == 'timestep':
            self.fortran.pe0_bcast_int(M.itt)
            self.time_step()
            self.time_goes_by()
          elif cmd == 'quit':
            stop = True
          elif cmd == 'snapshot':
            self.write_snap_cdf()
          elif cmd == 'average':
            self.write_average_cdf()
          elif cmd == 'read_restart':
            self.read_restart()
          elif cmd == 'write_restart':
            self.write_restart()
          elif cmd == 'user_defined':
            self.user_defined_signal()
          else:
              raise pyOMError(' wrong signal in mainloop')
       return

   def user_defined_signal(self):
       """ dummy routine, which can be overloaded
       """
       return
   
   def one_time_step(self):
     """
     enter a simple model time stepping loop, only for PE0
     """
     if not self.halt:
       M=self.fortran.main_module      
       M.itt = M.itt + 1
       self.itt_str.set('time step %i' % M.itt)
       self.time = M.itt*M.dt_tracer
       self.set_signal('timestep')
       self.fortran.pe0_bcast_int(M.itt)
       self.time_step()
       if numpy.mod(M.itt,self.aveint.get())  == 0 and self.enable_average.get()==1: self.do_average()
       if numpy.mod(M.itt,self.snapint.get()) == 0 and self.enable_snapshot.get()==1: self.do_snapshot()
       if numpy.mod(M.itt,self.plotint.get()) == 0 and  self.enable_plotting.get()==1 : self.prep_plot()
       self.time_goes_by()
       self.after(1,self.one_time_step)
     return

   def start(self):
      """ start integration
      """ 
      if self.halt == 1:
        self.halt=0
        self.print_text('\ncontinue model integration')
        self.one_time_step()
      return

   def stop(self):
      """ stop integration
      """ 
      if self.halt == 0:
        self.halt=1
        self.print_text('\nstopping model integration')
      return

   def do_average(self):
        text='\nwriting average at %12.4es solver itts=%i'% (self.time, self.fortran.main_module.congr_itts)
        self.print_text(text)
        self.set_signal('average')
        self.write_average_cdf()
        return
     
   def do_snapshot(self):
        """ writing a snapshot
        """
        text='\nwriting a snapshot at %12.4es solver itts=%i'% (self.time, self.fortran.main_module.congr_itts)
        self.print_text(text)
        self.set_signal('snapshot')
        self.write_snap_cdf()
        return
      
   def prep_plot(self):
        """ prepare to make a plot
        """
        M=self.fortran.main_module         # fortran module with model variables
        if hasattr(self,'make_plot'):
           text='\nplotting at %12.4es solver itts=%i'% (self.time, self.fortran.main_module.congr_itts)
           self.print_text(text)
           self.make_plot() 
           self._plotit()
        if not hasattr(self,'make_plot'):
           self.print_text('\ncannot plot: missing method make_plot')
        return

   def _plotit(self):
        """ make the actual plot in a canvas
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.backends.tkagg  as tkagg
        canvas = FigureCanvasAgg(self.figure)
        canvas.draw()
        if hasattr(self,'main_area_id'): self.main_area.delete(self.main_area_id)
        self.main_area_id=self.main_area.create_image(self.image_width/2,self.image_height/2,image=self._tkphoto)
        tkagg.blit(self._tkphoto, canvas.renderer._renderer, colormode=2)
        return

   
   def scale_both(self):
      """ scale button for snapshort and snapshot intervall identical
      """
      if self.snapint == self.plotint:
         self.plotint = Tkinter.IntVar(); self.plotint.set( self.snapint.get() )
         self.scale_bar.config(variable=self.plotint,state=Tkinter.NORMAL)
      else:
         self.scale_bar.config(variable=self.snapint,state=Tkinter.DISABLED)
         self.plotint = self.snapint
      return

   def print_text(self,s):
        """ print text string s in window
        """
        self.time_step_list.config(state=Tkinter.NORMAL)
        self.time_step_list.insert(Tkinter.END,s)
        self.time_step_list.yview(Tkinter.MOVETO,1)
        self.time_step_list.config(state=Tkinter.DISABLED)
        return
  
   def printplot(self):
        """ message box
        """
        tkMessageBox.showinfo("Info", "not yet implemented")
        return
   
   def quit(self):
        """ Quit message box
        """
        if tkMessageBox.askokcancel(title='Quit',message='Exit pyOM?'): 
            self.set_signal('quit')
            Tkinter.Frame.quit(self)
        return
   
   def reset(self):
        """ reset integration, show box
        """
        self.stop()
        self.set_initial_conditions()   
        self.fortran.calc_initial_conditions()
        self.prep_plot()
        return
      
   def read(self):
         """ to read restart open file dialog
         """
         #file=tkFileDialog.askopenfilename()
         #if file:
         #    self.halt=1
         #    self.print_text('\nreading restart file %s' %file)
         #    self.read_restart(file)
         self.halt=1
         self.set_signal('read_restart')
         self.read_restart()
         return
           
   def write(self):
         """ to write restart open file dialog
         """
         self.halt=1
         self.print_text('\nstopping model integration')
         #file=tkFileDialog.asksaveasfilename(initialfile='restart.dta')
         #if file:
         #   self.print_text('\nwriting restart to file %s' %file)
         #   self.write_restart(file)
         self.set_signal('write_restart')
         self.write_restart()
         return
    

   def config(self):
        if hasattr(self,'config_setup'):
           self.config_window=Tkinter.Toplevel()
           self.config_window.protocol('WM_DELETE_WINDOW', self.quit)
           self.config_frame = Tkinter.Frame(self.config_window)
           self.config_frame.pack()
           #Tkinter.Label(self.config_frame,text='Config').pack(side=Tkinter.TOP)
           self.config_setup() # add widgets to frame
        else:
           tkMessageBox.showinfo("Info", "method config_setup not found")
        return

   # rest is this shell
   
   def shell(self):
      if self.shell_on:
        self.shell_window.destroy()
        self.shell_on=False
      else:   
        self.shell_window=Tkinter.Toplevel()
        self.shell_window.protocol('WM_DELETE_WINDOW', self.quit)

        fo=("Times",12)
        self.display=ScrolledText.ScrolledText(self.shell_window,width=80,font=fo)
        self.display.pack()
      
        self.command = Tkinter.StringVar()
        self.command.set('')
        if not hasattr(self,'commandlist'):
           self.commandlist = ['']
           self.command_pointer = 0

        self.entry = Tkinter.Entry(self.display,textvariable=self.command,width=77,font=fo,bd=0)
        self.entry.config(highlightthickness=0)
        self.display.config(state=Tkinter.NORMAL)
        self.display.insert(Tkinter.END,'\n Welcome to %s \n\n'% self.pyOM_vstr)
        self.display.insert(Tkinter.END,'>>')
        self.display.see(Tkinter.END)
        self.display.window_create(Tkinter.END,window=self.entry)
        self.display.config(state=Tkinter.DISABLED)
      
        self.entry.bind(sequence="<Return>", func=self.command_process)
        self.entry.bind(sequence="<Up>",     func=self.command_up)
        self.entry.bind(sequence="<Down>",   func=self.command_down)
        self.entry.focus_set()
        self.entry.icursor(Tkinter.END)
        self.shell_on=True
        if not hasattr(self,'command_locals'): self.set_command_locals()
      return

   def set_command_locals(self):
     M=self.fortran.main_module         # fortran module with model variables
     import matplotlib.pyplot as plt
     self.command_locals=locals()
     return 

   def command_up(self,args):
      self.command_pointer = max(0,self.command_pointer-1)
      self.command.set(self.commandlist[ self.command_pointer ] )
      self.entry.icursor(Tkinter.END)
 
   def command_down(self,args):
      self.command_pointer = min(len(self.commandlist),self.command_pointer+1 )
      if self.command_pointer == len(self.commandlist):
       self.command.set('')
      else:
       self.command.set( self.commandlist[ self.command_pointer ] )
      self.entry.icursor(Tkinter.END)
      
   def command_process(self, args):
      import sys,StringIO
      self.sendToDisplay('>>'+self.command.get()+"\n")
      if self.command.get() != '':
         self.commandlist.append(self.command.get() )
         self.command_pointer = len(self.commandlist )
      b1=sys.stderr; b2=sys.stdout
      sys.stderr = StringIO.StringIO()
      sys.stdout = StringIO.StringIO()
      c= code.InteractiveInterpreter(self.command_locals)
      c.runsource(self.command.get() )
      self.sendToDisplay(sys.stderr.getvalue())
      self.sendToDisplay(sys.stdout.getvalue())
      sys.stderr=b1; sys.stdout=b2
      self.command.set('')
      return
   
   def sendToDisplay(self, string):
      self.display.config(state=Tkinter.NORMAL)
      self.display.insert(Tkinter.END+'-4 chars', string)
      self.display.see(Tkinter.END)
      self.display.config(state=Tkinter.DISABLED)
      return
   

