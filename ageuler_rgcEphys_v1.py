import datajoint as dj
import matplotlib.pyplot as plt
import scipy.ndimage as scimage
import fnmatch
import h5py
import hashlib
from itertools import chain
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

schema = dj.schema('ageuler_rgcEphys_test1',locals())

@schema
class Animal(dj.Manual):
    definition = """
    # Basic animal info

    animal_id					:varchar(20)   																# unique ID given to the animal
    ---
    species="mouse"				:enum("mouse","rat","zebrafish")											# animal species
    animal_line="PvCreAi9"		:enum("PvCreAi9","B1/6","ChATCre","PvCreTdT","PCP2TdT","ChATCreTdT","WT")	# transgnenetic animal line, here listed: mouse lines
    gender						:enum("M","F","unknown")													# gender
    date_of_birth				:date																		# date of birth
    """

@schema
class Experiment(dj.Manual):
    definition = """
    # Basic experiment info

    -> Animal

    exp_date	:date										# date of recording
    eye			:enum("R","L")								# left or right eye of the animal
    ---
    experimenter="tstadler"		:varchar(20)				# first letter of first name + last name = lrogerson/tstadler
    setup="2"					:tinyint unsigned			# setup 1-3
    amplifier="abraham"			:enum("abraham","nikodemus")# amplifiers abraham and nikodemus for setup 1
    preparation="wholemount"	:enum("wholemount","slice")	# preparation type of the retina
    dye="sulfrho"				:enum("sulfrho")			# dye used for pipette solution to image morphology of the cell
    path						:varchar(200)				# relative path of the experimental data folder
    """

@schema
class Cell(dj.Manual):
    definition="""
    # Single cell info

    -> Experiment

    cell_id		:tinyint unsigned	# unique ID given to each cell patched
    ---
    abs_x=0		:smallint			#absolute x coordinate from the sutter in integer precision
    abs_y=0		:smallint			#absolute y coordinate from the sutter in integer precision
    abs_z=0		:smallint			#absolute z coordinate from the sutter in integer precision
    rel_x=0		:smallint			#relative x coordinate from the sutter in integer precision
    rel_y=0		:smallint			#relative y coordinate from the sutter in integer precision
    rel_z=0		:smallint			#relative z coordinate from the sutter in integer precision
    folder		:varchar(200)		#relative folder path for the subexperiment
    morphology  : boolean           # morphology was recorded or not
    type        : varchar(200)      # putative RGC type
    """

@schema
class Recording(dj.Manual):
    definition="""
    # Information for a particular recording file

    -> Cell
    filename		: varchar(200) 		  						# name of the converted recording file
    ---
    rec_type                            : enum('intracell','extracell')                     # recording is an intra- or juxtacellular recordin
    stim_type							: enum('bw_noise','chirp','ds','on_off')	# type of the stimulus played during recording
    fs=10000							: int					  					# sampling rate of the recording
    ch_voltage="Vm_scaled AI #10"		: varchar(50)								# name of the voltage channel for this recording
    ch_trigger="LightTrig Blanking"		: varchar(50)								# name of the channel containing the light trigger signal


    """

@schema
class Spikes(dj.Computed):
    definition="""
    # Spike times in the Recording

    -> Recording
    ---
    spiketimes		: longblob				# array spiketimes (1,nSpikes) containing spiketimes  in sample points
    nspikes         : int                   # number of spikes detected
    rec_len			: int					# number of data points sampled

    """

    def _make_tuples(self,key):


            # fetch required data

            filename = (Recording() & key).fetch1['filename']
            ch_voltage = (Recording() & key).fetch1['ch_voltage']
            rec_type = (Recording() & key).fetch1['rec_type']

            cell_path = (Cell() & key).fetch1['folder']
            exp_path = (Experiment() & key).fetch1['path']

            # extract raw data for the given recording
            full_path = exp_path + cell_path + filename + '.h5'
            f = h5py.File(full_path,'r')

            ch_grp = f[ch_voltage] # get each channel group into hdf5 grp object
            keylist = [key for key in ch_grp.keys()] # get key within one group
            voltage_trace  = ch_grp[keylist[1]]['data'][:] # initialize as section_00
            for sec in range(2,len(keylist)):
                ch_sec_tmp = ch_grp[keylist[sec]]
                dset = ch_sec_tmp['data'][:] # get array
                voltage_trace = np.append(voltage_trace,dset)

            if rec_type == 'extracell':

                # determine threshold
                sigma = np.median(np.abs(voltage_trace)/.6745)
                thr = 5 * sigma

                # threshold signal
                tmp = np.array(voltage_trace)
                thr_boolean = [tmp > -thr]
                tmp[thr_boolean] = 0

                # detect spikes as threshold crossings
                tmp[tmp!=0]=1
                tmp = tmp.astype(int)
                tmp2 = np.append(tmp[1:len(tmp)],np.array([0],int))
                dif = tmp2-tmp

            elif rec_type == 'intracell':
                voltage_trace = rawdata[ch_voltage]
                d_voltage = np.array(np.diff(voltage_trace))
                sigma = np.median(np.abs(d_voltage + np.abs(min(d_voltage)))/(.6745))
                thr = sigma
                print('Threshold is', thr)
                tmp = np.array(d_voltage + np.abs(min(d_voltage)))
                thr_boolean = [tmp < thr]
                tmp[thr_boolean]=0
                tmp[tmp!=0]=1
                tmp = tmp.astype(int)
                tmp2 = np.append(tmp[1:len(tmp)],np.array([0],int))
                dif = tmp2-tmp

            else:
                print('Unknown recording type')

            spiketimes = np.where(dif==-1)[0]

            # insert
            self.insert1(dict(key, spiketimes=spiketimes, rec_len=len(voltage_trace),nspikes=len(spiketimes)))

    def plt_rawtrace(self):

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            ch_voltage, ch_trigger = (Recording() & key).fetch1['ch_voltage','ch_trigger']

            cell_path = (Cell() & key).fetch1['folder']
            exp_path = (Experiment() & key).fetch1['path']

            full_path = exp_path + cell_path + fname + '.h5'
            f = h5py.File(full_path,'r')
            ch_grp = f[ch_voltage]
            keylist = [key for key in ch_grp.keys()] # get key within one group
            voltage_trace  = ch_grp[keylist[1]]['data'][:] # initialize as section_00

            for sec in range(2,len(keylist)):
                ch_sec_tmp = ch_grp[keylist[sec]]
                dset = ch_sec_tmp['data'][:] # get array
                voltage_trace = np.append(voltage_trace,dset)

            ch_grp = f[ch_trigger]
            keylist = [key for key in ch_grp.keys()] # get key within one group
            trigger_trace  = ch_grp[keylist[1]]['data'][:] # initialize as section_00

            for sec in range(2,len(keylist)):
                ch_sec_tmp = ch_grp[keylist[sec]]
                dset = ch_sec_tmp['data'][:] # get array
                trigger_trace = np.append(trigger_trace,dset)

            plt.rcParams.update({'figure.figsize':(10,8),'xtick.labelsize': 16,'ytick.labelsize':16,'axes.labelsize':16,'axes.titlesize':20,
                         'savefig.transparent':False,'savefig.pad_inches':.2,'savefig.bbox':'tight','figure.subplot.hspace':.1})


            fig, axarr = plt.subplots(2,1,sharex = True)

            t = np.linspace(0,30,300000)
            axarr[0].set_title(fname)
            axarr[0].plot(t,voltage_trace[0:300000],'k',linewidth=1)
            axarr[0].set_ylabel('voltage [mV]',labelpad=10)

            axarr[1].plot(t,trigger_trace[0:300000],'k',linewidth=1)
            axarr[1].set_xlabel('time [s]')
            axarr[1].set_xticks([0,15,30])
            axarr[1].set_yticks([0,np.max(trigger_trace)])
            axarr[1].set_yticklabels(['0','1'])
            axarr[1].set_ylabel('Light Trigger',labelpad=20)

@schema
class Trigger(dj.Computed):
    definition="""
    ->Recording
    ---
    trigger_n	:longblob	# trigger times in sample points
    """

    def _make_tuples(self,key):
        # fetch required data

        fname = (Recording() & key).fetch1['filename']
        ch_trigger = (Recording() & key).fetch1['ch_trigger']

        cell_path = (Cell() & key).fetch1['folder']
        exp_path = (Experiment() & key).fetch1['path']

        # extract raw data for the given recording
        full_path = exp_path + cell_path + fname + '.h5'
        f = h5py.File(full_path,'r')

        ch_grp = f[ch_trigger] # get each channel group into hdf5 grp object
        keylist = [key for key in ch_grp.keys()] # get key within one group
        trigger_trace  = ch_grp[keylist[1]]['data'][:] # initialize as section_00
        for sec in range(2,len(keylist)):
            ch_sec_tmp = ch_grp[keylist[sec]]
            dset = ch_sec_tmp['data'][:] # get array
            trigger_trace = np.append(trigger_trace,dset)

        # get trigger times by diff

        tmp = np.array(trigger_trace)
        thr_boolean = [tmp < 1]
        tmp[thr_boolean] = 0
        tmp[tmp!=0]=1
        tmp2 = np.append(tmp[1:len(tmp)],[0])
        dif = tmp-tmp2
        trigger_n = np.where(dif==-1)[0]

        # insert
        self.insert1(dict(key, trigger_n=trigger_n))

@schema
class BWNoise(dj.Computed):
    definition="""
    # Fetch parameter set and stimulus array from filename

    -> Recording

    ---
    freq	:int										# stimulation frequency in Hertz
    delx	:int										# size of a pixel in x dimensions
    dely	:int										# size of a pixel in y dimensions
    mseq	:varchar(500)								# name of txt file containing mseq
    """
    @property
    def populated_from(self):
        return Recording() & dict(stim_type='bw_noise')

    def _make_tuples(self, key):

        #fetch required data

        fname = (Recording() & key).fetch1['filename']

        # extract stimulus parameter for the given recording

        if '5Hz' in fname:
            if '40um' in fname:
                if 'long' in fname:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_long'))
                else:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_official'))


            elif ('20um' in fname) or ('BCNoise' in fname):
                if 'long' in fname:
                    self.insert1(dict(key,freq=5,delx=20, dely=20,mseq='BWNoise_long'))
                else:
                    self.insert1(dict(key,freq=5,delx=20, dely=20,mseq='BWNoise_official'))
            else:
                if 'long' in fname:
                    self.insert1(dict(key,freq=5,delx=40, dely=40,mseq='BWNoise_long'))
                else:
                    self.insert1(dict(key,freq=5,delx=40, dely=40,mseq='BWNoise_official'))

        elif '20Hz' in fname:
            if '40um' in fname:
                if 'long' in fname:
                    self.insert1(dict(key,freq=20,delx=40,dely=40,mseq='BWNoise_long'))
                else:
                    self.insert1(dict(key,freq=20,delx=40,dely=40,mseq='BWNoise_official'))


            elif ('20um' in fname) or ('BCNoise' in fname):
                if 'long' in fname:
                    self.insert1(dict(key,freq=20,delx=20, dely=20,mseq='BWNoise_long'))
                else:
                    self.insert1(dict(key,freq=20,delx=20, dely=20,mseq='BWNoise_official'))
            else:
                if 'long' in fname:
                    self.insert1(dict(key,freq=20,delx=40, dely=40,mseq='BWNoise_long'))
                else:
                    self.insert1(dict(key,freq=20,delx=40, dely=40,mseq='BWNoise_official'))

        else:
            if '40um' in fname:
                if 'long' in fname:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_long'))
                else:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_official'))


            elif ('20um' in fname) or ('BCNoise' in fname):
                if 'long' in fname:
                    self.insert1(dict(key,freq=5,delx=20, dely=20,mseq='BWNoise_long'))
                else:
                    self.insert1(dict(key,freq=5,delx=20, dely=20,mseq='BWNoise_official'))
            else:
                if 'long' in fname:
                    self.insert1(dict(key,freq=5,delx=40, dely=40,mseq='BWNoise_long'))
                else:
                    self.insert1(dict(key,freq=5,delx=40, dely=40,mseq='BWNoise_official'))

@schema
class BWNoiseFrames(dj.Computed):
    definition="""
    # Stimulus frames for a given mseq

    ->BWNoise
    ---
    frames		: longblob		# array of stimulus frames
    stim_length	: int			# number of frames
    stim_dim_x	: int			# number of pixel rows
    stim_dim_y	: int			# number of pixel columns
    """

    def _make_tuples(self, key):

        stim_folder = "/notebooks/Notebooks/Stadler/stimulus_ana/"
        mseq = (BWNoise() & key).fetch1['mseq']
        # read stimulus information into np.array
        mseq_name = stim_folder + mseq
        Frames = []
        stimDim = []

        lines=open(mseq_name + '.txt').read().split('\n')

        params = lines[0].split(',')
        stimDim.append(int(params[0]))
        stimDim.append(int(params[1]))
        stimDim.append(int(params[2]))

        nB = stimDim[0]*stimDim[1]

        for l in range(1,len(lines)):
            split = lines[l].split(',')
            Frame = np.array(split).astype(int)
            Frames.append(Frame)
        Frames = Frames - np.mean(Frames,0)


        # insert data
        self.insert1(dict(key,frames=Frames, stim_length=stimDim[2], stim_dim_x=stimDim[0], stim_dim_y=stimDim[1]))

@schema
class STA(dj.Computed):
    definition="""
    # Calculate the spike-triggered ensemble for noise recording
    -> Recording
    ---
    sta    : longblob	# spike-triggered average
    """
    @property
    def populated_from(self):
        return Recording() & dict(stim_type='bw_noise')

    def _make_tuples(self,key):
        spikes = (Spikes() & key).fetch1['spiketimes']
        trigger = (Trigger() & key).fetch1['trigger_n']
        frames = (BWNoiseFrames() & key).fetch1['frames']
        (stim_length, stim_dim_x,stim_dim_y) = (BWNoiseFrames() & key).fetch1['stim_length','stim_dim_x','stim_dim_y']
        stim_freq = (BWNoise() & key).fetch1['freq']
        rec_len = (Spikes() & key).fetch1['rec_len']
        fs = (Recording() & key).fetch1['fs']

        stimInd = np.zeros(rec_len).astype(int)-1

        if len(trigger) != stim_length:
            print('Something went wrong with the trigger detection!')

        else:

            for n in range(len(trigger)-1):
                stimInd[trigger[n]:trigger[n+1]-1] += int(n+1)
            stimInd[trigger[len(trigger)-1]:trigger[len(trigger)-1]+(fs/stim_freq)-1] += int(len(trigger))


        deltat = 1000
        delta = int(deltat/.1)
        spikes = spikes[spikes > trigger[0]+delta]
        spikes = spikes[spikes < trigger[len(trigger)-1] + int(fs/stim_freq)-1]
        nspikes = len(spikes)
        k = 100

        ste = np.zeros([nspikes,(delta+1000)/k,stim_dim_x*stim_dim_y])
        for st in range(nspikes):
            for t in range(-1000,delta,k):
                ste[st,int((t+1000)/k),:] = np.array(frames[stimInd[spikes[st]-t]])

        sta = np.mean(ste,0)

        self.insert1(dict(key,sta=sta))

    def plt_rf(self):

        plt.rcParams.update({'figure.subplot.hspace':0,'figure.subplot.wspace':.3,'figure.figsize':(12,8),'axes.titlesize':16})

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            sta = (self & key).fetch1['sta']
            stimDim = (BWNoiseFrames() & key).fetch1['stim_dim_x','stim_dim_y']
            sta_smooth = scimage.filters.gaussian_filter(sta.reshape(sta.shape[0],stimDim[0],stimDim[1]),[0.2,.7,.7]) # reshape and smooth with a gaussian filter
            sta_norm = sta_smooth/np.std(sta_smooth,0)

            fig, axarr = plt.subplots(2,int(np.ceil(sta.shape[0]/20)))
            fig.subplots_adjust(hspace=.1,wspace=.1)

            if  (int(np.ceil(sta.shape[0])) % 20 == 0):
                ax = axarr.reshape(int(np.ceil(sta.shape[0]/10)))
            else:
                ax = axarr.reshape(int(np.ceil(sta.shape[0]/10))+1)
                im = ax[int(np.ceil(sta.shape[0]/10))].imshow(np.zeros([20,15]),cmap = plt.cm.Greys_r,interpolation='none',clim=(-1,1))
                ax[int(np.ceil(sta.shape[0]/10))].set_xticks([])
                ax[int(np.ceil(sta.shape[0]/10))].set_yticks([])

            tmp = 1

            with sns.axes_style(style = 'whitegrid'):

                for delt in range(0,sta.shape[0],10):


                        im = ax[delt/10].imshow(sta_norm[delt,:,:],
                                        cmap = plt.cm.coolwarm,clim = (-np.percentile(sta_norm,90),np.percentile(sta_norm,90)),interpolation='none')
                        ax[delt/10].set_title('$\Delta$ t = ' + str(-(delt-10)*10) + 'ms')
                        ax[delt/10].set_yticks([])
                        ax[delt/10].set_xticks([])
                        tmp += 1

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label('s.d. units',labelpad = 40,rotation=270)

                plt.suptitle('STA for different time lags\n' + fname,fontsize=16)


    def plt_contour(self,tau,x1,x2,y1,y2):

        plt.rcParams.update({'lines.linewidth':4})

        for key in self.project().fetch.as_dict:


            sta = (self & key).fetch1['sta']
            fname = key['filename']

            stimDim = (BWNoiseFrames() & key).fetch1['stim_dim_x','stim_dim_y']
            sta_smooth = scimage.filters.gaussian_filter(sta.reshape(sta.shape[0],stimDim[0],stimDim[1]),[0.2,.7,.7]) # reshape and smooth with a gaussian filter

            frame = int(10 - tau/10)

            fig_contour = plt.figure()
            plt.title('$\Delta$  t: ' + str(tau) + '\n' + fname)


            im = plt.imshow(sta_smooth[frame,:,:][x1:x2,y1:y2], interpolation='none',
                        cmap=plt.cm.Greys_r, extent=(y1,y2,x2,x1),origin='upper')
            cs = plt.contour(sta_smooth[frame,:,:][x1:x2,y1:y2],
                            extent=(y1,y2,x2,x1),cmap=plt.cm.coolwarm,origin='upper')

            cb = plt.colorbar(cs, extend='both',shrink=.8)
            cbaxes = fig_contour.add_axes([.25,.02, .5,.03]) #[left, bottom, width, height]
            cbi = plt.colorbar(im, orientation='horizontal',cax = cbaxes)

@schema
class ChirpParams(dj.Computed):
    definition="""
    -> Recording
    ---
    ntrials    : int	# how often was the stimulus looped
    """
    @property
    def populated_from(self):
        return Recording() & dict(stim_type='chirp')

    def _make_tuples(self,key):
        trigger = (Trigger() & key).fetch1['trigger_n']
        ntrials = int(np.floor(len(trigger)/2))
        self.insert1(dict(key,ntrials=ntrials))

@schema
class Chirp(dj.Computed):
    definition="""
    -> Recording
    -> ChirpParams
    ---
    psth_trials		: longblob		# psth per trial
    psth			: longblob		# average psth
    f_norm          : longblob      # spikes relative to trigger time, here a problem occurs because this is new????
    loop_duration_s	: double		# real duration of one stimulus loop
    qi				: double		# quality response index
    """

    @property
    def populated_from(self):
        return (Recording() & dict(stim_type='chirp'))


    def _make_tuples(self,key):
            trigger_n = (Trigger() & key).fetch1['trigger_n']
            spikes = (Spikes() & key).fetch1['spiketimes']
            ntrials = (ChirpParams() & key).fetch1['ntrials']
            fs = (Recording() & key).fetch1['fs']

            trigger_n = trigger_n.reshape(ntrials,2)

            true_loopDuration = []
            for trial in range(1,ntrials):
                true_loopDuration.append(trigger_n[trial,0] - trigger_n[trial-1,0])
            loopDuration_n =  np.ceil(np.mean(true_loopDuration)) # in sample points
            loopDuration_s = loopDuration_n/fs # in s


            f = []
            for trial in range(ntrials-1):
                f.append(np.array(spikes[(spikes>trigger_n[trial,0]) & (spikes<trigger_n[trial+1,0])]))
            f.append(np.array(spikes[(spikes>trigger_n[ntrials-1,0])& (spikes<trigger_n[ntrials-1,0]+ loopDuration_n)]))

            f_norm = []
            for trial in range(ntrials):
                f_norm.append(f[trial]-trigger_n[trial,0])

            T = int(loopDuration_s) # in s
            delT = .1 # in s
            nbins1 = T/delT

            psth = np.zeros(nbins1)
            psth_trials = []

            for trial in range(ntrials):

                psth_trials.append(np.histogram(f_norm[trial]/10000,nbins1,[0,T])[0])
                psth += psth_trials[trial]

                psth = psth/(delT*ntrials)

            R = np.array(psth_trials).transpose()
            QI_Chirp = np.var(np.mean(R,1))/np.mean(np.var(R,0)).astype(float)

            self.insert1(dict(key,psth_trials = np.array(psth_trials),psth = np.array(psth),f_norm = np.array(f_norm),loop_duration_s = loopDuration_s,qi=QI_Chirp))

    def plt_chirp(self):

        plt.rcParams.update({'figure.figsize':(10,8),'lines.linewidth':2})

        # define stimulus


        for key in self.project().fetch.as_dict:

            loopDuration_s = (self & key).fetch1['loop_duration_s']
            psth = (self & key).fetch1['psth']
            f_norm = (self & key).fetch1['f_norm']
            fname = (self & key).fetch1['filename']
            ntrials = int((ChirpParams() & key).fetch1['ntrials'])
            delT = .1


            ChirpDuration = 8                   # Time (s) of rising/falling chirp phase
            ChirpMaxFreq  = 8                   # Peak frequency of chirp (Hz)
            IntensityFrequency = 2               # freq at which intensity is modulated


            SteadyOFF     = 3.00                 # Time (s) of Light OFF at beginning at end of stimulus
            SteadyOFF2     = 2.00
            SteadyON      = 3.00                 # Time (s) of Light 100% ON before and after chirp
            SteadyMID     = 2.00                 # Time (s) of Light at 50% for steps

            Fduration     = 0.017               # Single Frame duration (s) -  ADJUST DEPENDING ON MONITOR
            #Fduration_ms  = 17.0                 # Single Frame duration (ms) - ADJUST DEPENDING ON MONITOR

            KK = ChirpMaxFreq / ChirpDuration # acceleration in Hz / s
            KK2 = IntensityFrequency

            StimDuration = SteadyOFF2+SteadyON+2*SteadyOFF+3*SteadyMID+2*ChirpDuration

            def stimulus():
                t = np.linspace(0,ChirpDuration,ChirpDuration/Fduration)
                Intensity0 = np.sin(3.141 * KK * np.power(t,2) ) * 127 + 127
                RampIntensity = 127*t/(ChirpDuration)
                Intensity1 = np.sin(2*3.141 * KK2 * t) * RampIntensity  + 127

                n_off = SteadyOFF/Fduration
                n_off2 = SteadyOFF2/Fduration
                n_on = SteadyON/Fduration
                n_midi = SteadyMID/Fduration
                n_chirp = ChirpDuration/Fduration

                t_on = n_off2
                t_off0 = n_off2+n_on
                t_midi0 = n_off2+n_on+n_off
                t_chirp0 = n_off2+n_on+n_off+n_midi
                t_midi1 = n_off2+n_on+n_off+n_midi+n_chirp
                t_chirp1 = n_off2+n_on+n_off+n_midi+n_chirp+n_midi
                t_midi2 = n_off2+n_on+n_off+n_midi+n_chirp+n_midi+n_chirp
                t_off1 = n_off2+n_on+n_off+n_midi+n_chirp+n_midi+n_chirp + n_midi

                tChirp = np.linspace(0,StimDuration,StimDuration/Fduration)
                chirp = np.zeros(len(tChirp))

                chirp[t_on:t_off0-1] = 255
                chirp[t_midi0:t_chirp0] = 127
                chirp[t_chirp0:t_midi1] = Intensity0
                chirp[t_midi1:t_chirp1] = 127
                chirp[t_chirp1:t_midi2-1] = Intensity1
                chirp[t_midi2:t_off1] = 127

                return tChirp,chirp


            T = int(loopDuration_s) # in s

            nbins1 = T/delT

            tPSTH = np.linspace(0,T,nbins1)

            fig, axarr = plt.subplots(3,1,sharex=True)
            plt.subplots_adjust(hspace=.7)

            for trial in range(ntrials):
                axarr[1].scatter(f_norm[trial]/10000,trial*np.ones([len(f_norm[trial]/10000)]),color='k') # scatter(tStar{trial},trial*ones(1,length(tStar{trial})),'b.')
                axarr[1].set_ylabel('# trial')

            axarr[1].set_ylim([0,ntrials])

            axarr[2].plot(tPSTH,psth,'k')
            axarr[2].set_ylabel('PSTH')
            axarr[2].set_yticks([0,max(psth)/2,max(psth)])
            axarr[2].set_xlabel('time [s]')

            (tChirp,chirp) = stimulus()

            axarr[0].plot(tChirp,chirp,'k')
            axarr[0].set_ylabel('stimulus intensity')
            axarr[0].set_yticks([0,127,250])

            axarr[0].set_title('Chirp\n' + fname)

@schema
class DSParams(dj.Computed):
    definition="""
    -> Recording
    ---
    nconditions=8 	: int	# number of directions bar was tested
    """

    @property
    def populated_from(self):
        return Recording() & dict(stim_type='ds')

    def _make_tuples(self,key):

        self.insert1(dict(key,nconditions=8))

@schema
class DS(dj.Computed):
    definition="""
    -> Recording
    -> DSParams
    ---
    hist		    : longblob    # number of spikes per direction sorted as 0° , 180°, 45°, 225°, ...
    hist_sorted	    : longblob    # number of spikes per direction sorted 0° - 315°
    dsi			    : double	  # direction selectivity index
    qi			    : double	  # quality response index
    """

    @property
    def populated_from(self):

        return Recording() & dict(stim_type='ds')

    def _make_tuples(self,key):

        # fetch data
        spikes = (Spikes() & key).fetch1['spiketimes']
        trigger_n = (Trigger() & key).fetch1['trigger_n']
        nconditions = (DSParams() & key).fetch1['nconditions']

        ntrials = int(len(trigger_n)/nconditions)

        deg = np.array([0,180,45,225,90,270,135,315]) #np.arange(0, 360, 360/nconditions)
        idx = np.array([0,4,6,2,5,1,7,3])

        true_loop_duration = []
        for trial in range(1,ntrials):
            true_loop_duration.append(trigger_n[trial*nconditions] - trigger_n[(trial-1)*nconditions])
        loop_duration_n =  np.ceil(np.mean(true_loop_duration)) # in sample points
        #loopDuration_s = loopDuration_n/10000 # in s


        spikes_trial = []
        spikes_normed = []
        hist = []
        hist_sorted = []

        for trial in range(ntrials-1):
            spikes_trial.append(np.array(spikes[(spikes>trigger_n[trial*nconditions]) & (spikes<trigger_n[(trial+1)*nconditions])]))
            spikes_normed.append(spikes_trial[trial]-trigger_n[trial*nconditions])
            hist.append(np.histogram(spikes_normed[trial],8,[0,loop_duration_n])[0])

            # sort by condition
            hist_sorted.append(hist[trial][idx])

        spikes_trial.append(np.array(spikes[(spikes > trigger_n[(ntrials-1)*nconditions])
                                            & (spikes < trigger_n[(ntrials-1)*nconditions]+loop_duration_n)]))
        spikes_normed.append(spikes_trial[ntrials-1]-trigger_n[(ntrials-1)*nconditions])
        hist.append(np.histogram(spikes_normed[ntrials-1],8,[0,loop_duration_n])[0])
        hist_sorted.append(hist[ntrials-1][idx])

        hist_sum = np.sum(hist,0)
        r_p = np.max(hist_sum)
        idx_p = np.where(hist_sum == r_p)[0][0]
        d_p = deg[idx_p]
        if (idx_p % 2) == 0:
            d_n = deg[idx_p + 1]
            r_n = hist_sum[idx_p + 1]
        else:
            d_n = deg[idx_p - 1]
            r_n = hist_sum[idx_p - 1]
        dsi = (r_p - r_n)/(r_p + r_n)

        R = np.array(hist).transpose()
        qi = np.var(np.mean(R,1))/np.mean(np.var(R,0))

        self.insert1(dict(key, hist = np.array(hist),hist_sorted = np.array(hist_sorted),dsi = dsi,qi = qi)) # spikes_trial = np.array(spikes_trial),spikes_normed = np.array(spikes_normed),hist = np.array(hist),hist_sorted = np.array(hist_sorted),dsi = dsi, qi = qi

    def plt_ds(self):

        plt.rcParams.update({'xtick.labelsize': 16,'ytick.labelsize':16,'axes.labelsize':16,'axes.titlesize':20,'figure.figsize':(10,8)})

        for key in self.project().fetch.as_dict:

            hist = (self & key).fetch1['hist']
            nconditions = (DSParams() & key).fetch1['nconditions']
            fname = key['filename']
            qi = (self & key).fetch1['qi']
            dsi = (self & key).fetch1['dsi']

            deg = np.array([0,180,45,225,90,270,135,315])


            with sns.axes_style('whitegrid'):
                fig = plt.figure()
                ax = plt.axes(polar=True,axisbg='white')
                width=.2
                rads = np.radians(deg)-width/2
                counts = np.mean(hist,0)
                plt.bar(rads,counts,width=width,facecolor='k')

                ycounts = [round(max(counts)/2),round(max(counts))]
                ax.set_theta_direction(-1)
                ax.set_theta_offset(np.pi/2)
                ax.set_yticks(ycounts)
                ax.grid(color = 'k',linestyle='--')
                ax.set_title('Directional Tuning\n' + fname,y = 1.1)
                ax.annotate('QI: ' + str("%.2f" % round(qi,2)),xy = (.8,.3),xycoords='figure fraction',size=16)
                ax.annotate('DSI: ' + str("%.2f" % round(dsi,2)),xy = (.85,.4),xycoords='figure fraction',size=16)

    def plt_ds_traces(self):

        plt.rcParams.update({'xtick.labelsize': 16,'ytick.labelsize':16,'axes.labelsize':16,'axes.titlesize':20,'figure.figsize':(10,8)})

        for key in self.project().fetch.as_dict:

            fname = (Recording() & key).fetch1['filename']
            ch_trigger = (Recording() & key).fetch1['ch_trigger']
            ch_voltage = (Recording() & key).fetch1['ch_voltage']

            cell_path = (Cell() & key).fetch1['folder']
            exp_path = (Experiment() & key).fetch1['path']

            # extract raw data for the given recording
            full_path = exp_path + cell_path + fname + '.h5'
            f = h5py.File(full_path,'r')

            ch_grp_trig = f[ch_trigger] # get each channel group into hdf5 grp object
            keylist = [key for key in ch_grp_trig.keys()] # get key within one group
            trigger_trace  = ch_grp_trig[keylist[1]]['data'][:] # initialize as section_00
            for sec in range(2,len(keylist)):
                ch_sec_tmp = ch_grp_trig[keylist[sec]]
                dset = ch_sec_tmp['data'][:] # get array
                trigger_trace = np.append(trigger_trace,dset)

            ch_grp_v = f[ch_voltage] # get each channel group into hdf5 grp object
            keylist = [key for key in ch_grp_v.keys()] # get key within one group
            voltage_trace  = ch_grp_v[keylist[1]]['data'][:] # initialize as section_00
            for sec in range(2,len(keylist)):
                ch_sec_tmp = ch_grp_v[keylist[sec]]
                dset = ch_sec_tmp['data'][:] # get array
                voltage_trace = np.append(voltage_trace,dset)

            trigger_n = (Trigger() & key).fetch1['trigger_n']

            # stimulus parameter
            fs = (Recording() & key).fetch1['fs']
            t_off  = 2*fs # in s * fs
            t_on = 2*fs # in s

            nconditions = (DSParams() & key).fetch1['nconditions']
            ntrials = int(len(trigger_n)/nconditions)
            idx = np.array([0,4,6,2,5,1,7,3])
            deg = np.arange(0, 360, 360/8).astype(int)

            true_loop_duration = []
            for trial in range(1,ntrials):
                true_loop_duration.append(trigger_n[trial*nconditions] - trigger_n[(trial-1)*nconditions])
            loop_duration_n =  np.ceil(np.mean(true_loop_duration))


            stim = np.zeros(trigger_trace.shape)

            for i in range(len(trigger_n)):
                stim[trigger_n[i]:trigger_n[i]+t_on] = 1

            v_trace_trial = []
            stim_trial = []
            for i in range(len(trigger_n)):
                v_trace_trial.append(np.array(voltage_trace[trigger_n[i]:trigger_n[i]+t_on+t_off]))
                stim_trial.append(np.array(stim[trigger_n[i]:trigger_n[i]+t_on+t_off]))

            plt.rcParams.update({'figure.subplot.hspace':.1,'figure.figsize':(20,8)})
            N = len(v_trace_trial)
            fig1, axarr = plt.subplots(int(N/nconditions)+1,nconditions,sharex = True,sharey = True) # len(trigger_n)
            for i in range(N+nconditions):
                rowidx = int(np.floor(i/nconditions))
                colidx = int(i-rowidx*nconditions)
                if rowidx == 0:
                    axarr[rowidx,colidx].plot(stim_trial[i]*np.max(v_trace_trial)-np.max(v_trace_trial),'k')
                    axarr[rowidx,colidx].set_xticks([])
                    axarr[rowidx,colidx].set_yticks([])
                else:
                    axarr[rowidx,colidx].plot(v_trace_trial[i-nconditions],'k')
                    axarr[rowidx,colidx].set_xticks([])
                    axarr[rowidx,colidx].set_yticks([])
            plt.suptitle(fname,fontsize=20)

            rec_type = (Recording() & key).fetch1['rec_type']
            # Heatmap
            if rec_type == 'intracell':
                arr = np.array(v_trace_trial)
                arr = arr.reshape(ntrials,nconditions,arr.shape[1])

                for trial in range(ntrials):
                    arr[trial,:,:] = arr[trial,:,:][idx]

                l = []
                for cond in range(nconditions):
                    for trial in range(ntrials):
                        l.append(arr[trial,cond,:])


                fig2,ax = plt.subplots()

                intensity = np.array(l).reshape(ntrials,nconditions,len(l[0]))

                column_labels = np.linspace(0,4,5)
                row_labels = deg.astype(int)

                plt.pcolormesh(np.mean(intensity,0),cmap = plt.cm.coolwarm)
                cax = plt.colorbar()
                cax.set_label('voltage [mV]',rotation=270,labelpad=50)
                plt.xlabel('time [s]')
                plt.ylabel('direction [deg]')
                plt.title('Average membrane potential')


                ax.set_xticks(np.linspace(0,len(l[0]),5), minor=False)
                ax.set_yticks(np.arange(intensity.shape[1]) + .5, minor=False)

                ax.invert_yaxis()
                ax.xaxis.set_ticks_position('bottom')

                ax.set_xticklabels(column_labels, minor=False)
                ax.set_yticklabels(row_labels, minor=False)
            else:
                fig2,ax = plt.subplots()

class Helpers:


    def fileScan(dataDirectory = '/notebooks/Data',fileTypes = ['abf','ini','h5','smh','smp']):
        fileLocation_pathlist = list(chain(*[findFileType('*.' + suffix, dataDirectory) for suffix in fileTypes]))
        fileLocation_table = locationToTable(fileLocation_pathlist)

        return fileLocation_table

    def findFileType(fileType,directory): # Type: '*.ini'
        # Find .ini files in all folders and subfolders of a specified directory
        fileLocation = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(directory)
        for f in fnmatch.filter(files,fileType)]
        return fileLocation

    def parseFileLocation(fileLocation,targetString='/'):
        # Extract useful information from directory path
        for fileType in list(fileLocation.keys()):
            backslash = [m.start() for m in re.finditer(targetString, entry)]
            headerFiles[entry].loc['surname'] = ['string',entry[backslash[-4]+1:backslash[-3]]]
            headerFiles[entry].loc['date'] = ['string',entry[backslash[-3]+1:backslash[-2]]]
            headerFiles[entry].loc['nExperiment'] = ['string',entry[backslash[-2]+1:backslash[-1]]]
        return headerFiles

    def readSHA1(fileLocation):
        # Find SHA-1 for file at file location
        BLOCKSIZE = 65536
        hasher = hashlib.sha1()
        with open(fileLocation, 'rb') as afile:
            buf = afile.read(BLOCKSIZE)
            while len(buf) > 0:
                hasher.update(buf)
                buf = afile.read(BLOCKSIZE)
        return hasher.hexdigest()

    def locationToTable(fileLocation_pathlist,targetString='/',topDirectory='Data/'):
        # Specification of dataframe in which to store information about files
        tableColumns = ['Surname', 'Date', 'Experiment', 'Subexpr', 'Filename', 'Filetype', 'Path', 'SHA1']
        fileLocation_table = pd.DataFrame(columns=tableColumns)

        for itx in range(len(fileLocation_pathlist)):
            path = fileLocation_pathlist[itx]
            # Find filepath within top directory, typically the 'Data/' folder
            subDirectory = path[path.find(topDirectory)+len(topDirectory):]

            # Find location of backslashes, which demarcate folders
            backslash = [m.start() for m in re.finditer(targetString, subDirectory)]

            # Extract folder and file names
            file = subDirectory[backslash[-1]+1:]
            fileName = file[:file.find('.')]
            fileType = file[file.find('.'):]
            Surname = subDirectory[:backslash[0]]
            Date = subDirectory[backslash[0]+1:backslash[1]]

            Experiment = np.nan
            Subexpr = np.nan
            if len(backslash) > 2:
                Experiment = subDirectory[backslash[1]+1:backslash[2]]
            if len(backslash) > 3:
                Subexpr = subDirectory[backslash[2]+1:backslash[3]]

            SHA1 = np.nan
            # SHA1 = readSHA1(path)

            fileEntry = [Surname,Date,Experiment,Subexpr,fileName,fileType,path,SHA1]
            fileLocation_table.loc[itx] = fileEntry

        return fileLocation_table

    def addEntry(animal_id,gender,date_of_birth,exp_date,eye,cell_id,morph,cell_type,data_folder,filename,rec_type,ch_voltage):
    """

    :param animal_id: str 'ZK0-yyyy-mm-dd'
    :param gender: str 'F' or 'M'
    :param date_of_birth: str 'yyyy-mm-dd'
    :param exp_date: str 'yyyy-mm-dd'
    :param eye: str 'R' or 'L'
    :param cell_id: int 1-16
    :param morph: boolean
    :param type: str 'putative cell type'
    :param data_folder: str '/notebooks/Data_write/Data/Stadler/'
    :param filename: str 'BWNoise'
    :return: adds the given recording to the mysql schema 'ageuler_rgcEphys'
    """
    from schema import Animal,Experiment,Cell,Recording
    A = Animal()
    E = Experiment()
    C = Cell()
    R = Recording()
    try:
        A.insert1({'animal_id':animal_id,'gender':gender,'date_of_birth':date_of_birth})
    except Exception as e1:
        print('Animal already is in db')
    try:
        exp_path = data_folder + exp_date + '/' + eye + '/'
        E.insert1({'animal_id':animal_id,'exp_date':exp_date,'eye':eye,'path':exp_path})

    except Exception as e2:
        print('Experiment already in db')
    try:
        subexp_path = str(cell_id) + '/'
        C.insert1({'animal_id':animal_id,'exp_date':exp_date,'eye':eye,'cell_id':cell_id,'folder':subexp_path,'morphology':morph,'type':cell_type})
    except Exception as e3:
        print('Cell already in db')
    try:
        if 'BWNoise' in filename:
            #fname = 'C' + str(cell_id) + '_' + filename
            R.insert1({'animal_id':animal_id,'exp_date':exp_date,'eye':eye,'cell_id':cell_id,'filename':filename,'stim_type':'bw_noise','rec_type':rec_type})
        if 'Chirp' in filename:
            #fname = 'C' + str(cell_id) + '_' + filename
            R.insert1({'animal_id':animal_id,'exp_date':exp_date,'eye':eye,'cell_id':cell_id,'filename':filename,'stim_type':'chirp','rec_type':rec_type})
        if 'DS' in filename:
            #fname = 'C' + str(cell_id) + '_' + filename
            R.insert1({'animal_id':animal_id,'exp_date':exp_date,'eye':eye,'cell_id':cell_id,'filename':filename,'stim_type':'ds','rec_type':rec_type})
        if 'ON' in filename:
            #fname = 'C' + str(cell_id) + '_' + filename
            R.insert1({'animal_id':animal_id,'exp_date':exp_date,'eye':eye,'cell_id':cell_id,'filename':filename,'stim_type':'on_off','rec_type':rec_type})
    except Exception as e4:
        print(e4)
        print('You already added this entry or the stimulus type was unknown')