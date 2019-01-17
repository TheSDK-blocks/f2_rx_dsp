#f2_dsp class 
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 14.11.2018 11:44
import numpy as np
import scipy.signal as sig
import tempfile
import subprocess
import shlex
import time

from thesdk import *
from f2_util_classes import *
from f2_decimator import *
import signal_generator_802_11n as sg80211n


class f2_rx_dsp(verilog,thesdk):
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__

    def __init__(self,*arg): 
        self.proplist = [ 'Rs', 
                          'Rs_dsp', 
                          'Rxantennas',
                          'Users',
                          'dsp_decimator_scales',      # Scales for the rx decimator chain
                          'dsp_decimator_cic3shift',   # Left-shift for the decimator cic integrator
                          'rx_output_mode'
                          ];    
        self.Rs = 160e6;                 # sampling frequency
        self.Rs_dsp=20e6
        #These are fixed
        self.Rxantennas=4
        self.Users=4
        self.Rxindex=0
        self.Userindex=0
        self.iptr_A = IO() 
        self.model='py';                 #can be set externally, but is not propagated
        self.dsp_decimator_model='py'    #Used only for python model byt can be set for testing
        self.dsp_decimator_scales=[1,1,1,1]
        self.dsp_decimator_cic3shift=0
        self.rtldiscard=50
        self.rx_output_mode=1
        #self.dspmode='local';              # [ 'local' | 'cpu' ]  
        self.par= False                  #by default, no parallel processing
        self.queue= []                   #by default, no parallel processing

        self.DEBUG= False
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;
        #What is outputted to these 4 ports is selected with modes
        self.iptr_A.Data=[IO() for _ in range(self.Rxantennas)]
        self._io_ofifo=iofifosigs(**{'users':self.Users})
        self.preserve_iofiles='False'
        self.init()

    def init(self):
        self.def_verilog()

        #Add decimator
        self.decimator=[ f2_decimator() for i in range(self.Rxantennas)]
        for i in range(self.Rxantennas):
            self.decimator[i].Rs_high=self.Rs
            self.decimator[i].Rs_low=self.Rs_dsp
            self.decimator[i].model=self.dsp_decimator_model
            self.decimator[i].iptr_A=self.iptr_A.Data[i]
            self.decimator[i].scales=self.dsp_decimator_scales
            self.decimator[i].cic3shift=self.dsp_decimator_cic3shift
            self.decimator[i].init()

        self.mode=self.decimator[0].mode
        self._vlogmodulefiles =list(['clkdiv_n_2_4_8.v' ,'AsyncResetReg.v'])
        self._vlogparameters=dict([ ('g_Rs_high',self.Rs), ('g_Rs_low',self.Rs_dsp), 
            ('g_scale0',self.dsp_decimator_scales[0]),  
            ('g_scale1',self.dsp_decimator_scales[1]),  
            ('g_scale2',self.dsp_decimator_scales[2]),  
            ('g_scale3',self.dsp_decimator_scales[3]),
            ('g_cic3shift', self.dsp_decimator_cic3shift), 
            ('g_mode',self.mode),
            ('g_user_index', 0),
            ('g_antenna_index', 0),
            ('g_rx_output_mode', self.rx_output_mode), 
            ('g_input_mode', 0),
            ('g_adc_fifo_lut_mode' ,2) 
            ])
        
    def run(self,*arg):
        if len(arg)>0:
            self.par=True      #flag for parallel processing
            self.queue=arg[0]  #multiprocessing.queue as the first argument
        if self.model=='py':
            self.process_input()
        else: 
          self.write_infile()
          self.run_verilog()
          self.read_outfile()

    def process_input(self):
        #Could use parallel forloop here
        [ self.decimator[i].run() for i in range(self.Rxantennas) ]

        #Run decimatiors in parallel
        #Split in two phases: First, get the channel estimates
        # l=0
        # que=[]
        # proc=[]
        # for i in range(self.Rxantennas):
        #     self.decimator[i].init()
        #     que.append(multiprocessing.Queue())
        #     proc.append(multiprocessing.Process(target=self.decimator[i].run, args=(que[l],)))
        #     proc[l].start()
        #     l += 1 
        #         
        # #Collect results for dsps
        # l=0
        # for i in range(self.Rxantennas):
        #     for k in range(self.Users):
        #         self._decimated.Data[k].Data=que[l].get()
        #     proc2[l].join()
        #     l+=1
        
        #Each of the RX's provide self.Users datastreams
        ####################
        ### At this point we should add the beamforming and the demodulation
        ### BUT CURRENTLY WE DO NOT:
        ####################

        self.print_log(type='W', msg='Discarded %i zero samples to remove possibble initial transients in symbol sync.' %(self.rtldiscard))
        #Array (antennas,time,users) of decimated datsrteams  
        decimated= [ self.decimator[i]._Z.Data.reshape(-1,1)[self.rtldiscard::,0].reshape(-1,1)@np.ones((1,self.Users)) for i in range(self.Rxantennas)]
        sumuserstream=np.sum(decimated,0)
        seluser=[ decimated[i][:,self.Userindex].reshape(-1,1) for i in range(self.Rxantennas)]
        selrx=[ decimated[self.Rxindex][:,i].reshape(-1,1) for i in range(self.Users)]
        selrxuser=decimated[self.Rxindex][:,self.Userindex].reshape(-1,1)
         
        if (self.rx_output_mode==0):
            self.print_log(type='I', msg="Applying RX ouput mode %s - Bypass" %(self.rx_output_mode) )
            #Bypass the sampling rate is NOT reduced
            for k in range(self.Rxantennas):
                self._io_ofifo.data[k].udata.Data=self.iptr_A.Data[k].Data.reshape(-1,1)
                self._io_ofifo.data[k].uindex.Data=k*np.ones_like(self._io_ofifo.data[k].udata.Data[0].shape[0])
                self._io_ofifo.rxindex.Data=0*np.ones_like(self._io_ofifo.data[k].udata.Data[0].shape[0])
        elif (self.rx_output_mode==1):
            self.print_log(type='I', msg="Applying RX ouput mode %s - User %s selected from all RX's" %(self.rx_output_mode, self.Userindex) )
            for k in range(self.Rxantennas):
                self._io_ofifo.data[k].udata.Data=seluser[k].reshape(-1,1)
                self._io_ofifo.data[k].uindex.Data=self.Userindex*np.ones_like(self._io_ofifo.data[k].udata.Data[0].shape[0])
                self._io_ofifo.rxindex.Data=0*np.ones_like(self._io_ofifo.data[k].udata.Data[0].shape[0])
        elif (self.rx_output_mode==2):
            self.print_log(type='I', msg="Applying RX ouput mode %s - RX %s selected for all users's" %(self.rx_output_mode, self.Rxindex) )
            for k in range(self.Users):
                self._io_ofifo.data[k].udata.Data=selrx[k].reshape(-1,1)
                self._io_ofifo.data[k].uindex.Data=k*np.ones_like(self._io_ofifo.data[k].udata.Data[0].shape[0])
                self._io_ofifo.data[k].rxindex.Data=self.Rxindex*np.ones_like(self._io_ofifo.data[k].udata.Data[0].shape[0])
        elif (self.rx_output_mode==3):
            self.print_log(type='I', msg="Applying RX ouput mode %s - RX %s and user %s selected to ouput index 0" %(self.rx_output_mode, self.Rxindex, self.Userindex) )
            for k in range(self.Users):
                if k==0:
                    self._io_ofifo.data[k].udata.Data=selrxuser.reshape(-1,1)
                    self._io_ofifo.data[k].uindex.Data=self.Userindex
                    self._io_ofifo.rxindex.Data=self.Rxindex
                else:
                    self._io_ofifo.data[k].udata.Data=np.zeros_like(selrxuser.reshape(-1,1))
                    self._io_ofifo.data[k].uindex.Data=self.Userindex*np.ones_like(self._io_ofifo.data[k].udata.Data[0].shape[0])

                    self._io_ofifo.rxindex.Data=self.Rxindex*np.ones_like(self._io_ofifo.data[k].udata.Data[0].shape[0])
        elif (self.rx_output_mode==4):
            self.print_log(type='F', msg="RX ouput mode %s - User data streaming in serial manner is no longer supported " %(self.rx_output_mode) )

        elif (self.rx_output_mode==5):
            self.print_log(type='F', msg="RX ouput mode %s - User data is streamed out in time-interleaved indexed order from four DSP outputs is no longer supported" %(self.rx_output_mode) )

        elif (self.rx_output_mode==6):
            self.print_log(type='I', msg="Applying RX ouput mode %s - Summed data is streamed out. Output position index is user index" %(self.rx_output_mode) )
            for k in range(self.Users):
                self._io_ofifo.data[k].udata.Data=sumuserstream[:,k].reshape(-1,1)
                self._io_ofifo.data[k].uindex.Data=k*np.ones_like(self._io_ofifo.data[k].udata.Data[0].shape[0])
                self._io_ofifo.rxindex.Data=0*np.ones_like(self._io_ofifo.data[k].udata.Data[0].shape[0])
        else:
            #Bypass
            self.print_log(type='I', msg="Applying RX ouput mode %s - Bypass" %(self.rx_output_mode) )
            for k in range(self.Rxantennas):
                self._io_ofifo.data[k].udata.Data=self.iptr_A.Data[k].Data.reshape(-1,1)
                self._io_ofifo.data[k].uindex.Data=k*np.ones_like(self._io_ofifo.data[k].udata.Data[0].shape[0])
                self._io_ofifo.rxindex.Data=0*np.ones_like(self._io_ofifo.data[k].udata.Data[0].shape[0])

    def distribute_result(self,result):
        for k in range(self.Users):
            if self.par:
                self.queue.put(result[:,k].reshape(-1,1))
            self._io_ofifo.data[k].udata.Data=result[:,k].reshape((-1,1))
    
    def write_infile(self):
        rndpart=os.path.basename(tempfile.mkstemp()[1])
        if self.model=='sv':
            self._infile=self._vlogsimpath +'/A_' + rndpart +'.txt'
            self._outfile=self._vlogsimpath +'/Z_' + rndpart +'.txt'
        elif self.model=='vhdl':
            pass
        else:
            pass
        try:
          os.remove(self._infile)
        except:
          pass
        for i in range(self.Users):
            if i==0:
                indata=self.iptr_A.Data[i].Data.reshape(-1,1)
            else:
                indata=np.r_['1',indata,self.iptr_A.Data[i].Data.reshape(-1,1)]

        fid=open(self._infile,'wb')
        np.savetxt(fid,indata.view(float),fmt='%i', delimiter='\t')
        fid.close()

    def read_outfile(self):
        fid=open(self._outfile,'r')
        fromfile = np.loadtxt(fid,dtype=complex,delimiter='\t')
        #Of course it does not work symmetrically with savetxt
        for i in range(self.Users):
            if i==0:
                out=np.zeros((fromfile.shape[0],int(fromfile.shape[1]/2)),dtype=complex)
                out[:,i]=(fromfile[:,2*i]+1j*fromfile[:,2*i+1]) 
            else:
                out[:,i]=(fromfile[:,2*i]+1j*fromfile[:,2*i+1])
            maximum=np.amax([np.abs(np.real(out[:,i])), np.abs(np.imag(out[:,i]))])
            str="Output signal range is %i" %(maximum)
            self.print_log(type='I', msg=str)
        fid.close()
        #os.remove(self._outfile)
        #Copy the value to multiple outputs for users
        self.distribute_result(out)
  
