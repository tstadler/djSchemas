import datajoint as dj
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as scimage
import scipy.signal as scignal

schema = dj.schema('ageuler_rgcEphys_test3',locals())

@schema
class Animal(dj.Manual):
    definition = """
    # Basic animal info

    animal_id					:varchar(20)   																# unique ID given to the animal
    ---
    species="mouse"				:enum("mouse","rat","zebrafish")											# animal species
    animal_line="PvCreAi9"		:enum("PvCreAi9","B1/6","ChATCre","PvCreTdT","PCP2TdT","ChATCreTdT","WT")	# transgnenetic animal line, here listed: mouse lines
    sex 						:enum("M","F","unknown")													# gender
    date_of_birth				:date																		# date of birth
    """

@schema
class Experiment(dj.Manual):
    definition = """
    # Basic experiment info

    -> Animal

    exp_date	:date										# date of recording
    eye			:enum("R","L","unknown")					# left or right eye of the animal
    ---
    experimenter="tstadler"		:varchar(20)				            # first letter of first name + last name = mroman/lrogerson/tstadler
    setup="2"					:tinyint unsigned			            # setup 1-3
    amplifier="abraham"			:enum("abraham","nikodemus","ehrlich")  # amplifiers abraham and nikodemus for setup 1
    preparation="wholemount"	:enum("wholemount","slice")	            # preparation type of the retina
    dye="sulforho"				:enum("sulforho")			            # dye used for pipette solution to image morphology of the cell
    path						:varchar(200)				            # relative path of the experimental data folder
    """

@schema
class Cell(dj.Manual):
    definition="""
    # Single cell info

    -> Experiment

    cell_id		:tinyint unsigned	# unique ID given to each cell patched on that day in that eye
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
            fs = (Recording() & key).fetch1['fs']

            cell_path = (Cell() & key).fetch1['folder']
            exp_path = (Experiment() & key).fetch1['path']


            full_path = exp_path + cell_path + filename + '.h5'

            # extract raw data for the given recording

            f = h5py.File(full_path,'r')

            ch_grp = f[ch_voltage] # get each channel group into hdf5 grp object
            keylist = [key for key in ch_grp.keys()] # get key within one group
            voltage_trace  = ch_grp[keylist[1]]['data'][:] # initialize as section_00

            for sec in range(2,len(keylist)):
                ch_sec_tmp = ch_grp[keylist[sec]]
                dset = ch_sec_tmp['data'][:] # get array
                voltage_trace = np.append(voltage_trace,dset)

            # filter signal

            wn0 = 45 / fs
            wn1 = 55 / fs
            b, a = scignal.butter(2, [wn0, wn1], 'bandstop',
                                  analog=False)  # second order, critical frequency, type, analog or digital
            voltage_trace = scignal.filtfilt(b, a, voltage_trace)

            # spike detection


            if rec_type == 'extracell':

                # determine threshold

                sigma = np.median(np.abs(voltage_trace) / .6745)
                thr = 5 * sigma
                print('Threshold is -', thr, 'mV')

                # threshold signal
                tmp = np.array(voltage_trace)
                thr_boolean = [tmp > -thr]
                tmp[thr_boolean] = 0

                # detect spiketimes as threshold crossings
                tmp[tmp != 0] = 1
                tmp = tmp.astype(int)
                tmp2 = np.append(tmp[1:len(tmp)], np.array([0], int))
                dif = tmp2 - tmp

                spiketimes = np.where(dif == -1)[0]
                print('Number of spikes: ', len(spiketimes))

            if rec_type == 'intracell':

                sigma = np.median(np.abs(voltage_trace + np.abs(min(voltage_trace)))) / .6745

                d_voltage = np.append(np.array(np.diff(voltage_trace)), np.array(0))
                d_sigma = np.median(np.abs(d_voltage + np.abs(min(d_voltage))) / (.6745))

                tmp = np.array(voltage_trace + np.abs(min(voltage_trace)))
                d_tmp = np.array(d_voltage + np.abs(min(d_voltage)))

                tmp[tmp < sigma] = 0
                d_tmp[d_tmp < d_sigma] = 0
                tmp[tmp > sigma] = 1
                d_tmp[d_tmp > d_sigma] = 1

                tmp = tmp.astype(int)
                d_tmp = d_tmp.astype(int)
                tmp2 = np.append(tmp[1:len(tmp)], np.array([0], int))
                d_tmp2 = np.append(d_tmp[1:len(d_tmp)], np.array([0], int))

                dif = tmp2 - tmp
                d_dif = d_tmp2 - d_tmp

                spiketimes = np.where(d_dif == -1)[0]
                print('Number of spikes: ', len(spiketimes))

            rec_len = len(voltage_trace)

            start = 0
            end = 30

            fig_v = self.show_spiketimes(voltage_trace, spiketimes, start, end, fs)

            display(fig_v)

            adjust0 = bool(int(input('Adjust threshold? (Yes: 1, No: 0): ')))
            plt.close(fig_v)

            if adjust0:

                if rec_type == 'extracell':

                    adjust1 = True

                    while adjust1:
                        pol = bool(int(input('y-axis switch? (Yes: 1, No: 0): ')))
                        alpha = int(input('Scale factor for threshold: '))

                        # determine threshold

                        thr = alpha * sigma

                        if pol:
                            print('Threshold is', thr, 'mV')
                            # threshold signal
                            tmp = np.array(voltage_trace)
                            thr_boolean = [tmp < thr]
                            tmp[thr_boolean] = 0

                            # detect spiketimes as threshold crossings
                            tmp[tmp != 0] = 1
                        else:
                            print('Threshold is -', thr, 'mV')
                            # threshold signal
                            tmp = np.array(voltage_trace)
                            thr_boolean = [tmp > -thr]
                            tmp[thr_boolean] = 0

                            tmp[tmp != 0] = 1

                        tmp = tmp.astype(int)
                        tmp2 = np.append(tmp[1:len(tmp)], np.array([0], int))
                        dif = tmp2 - tmp

                        spiketimes = np.where(dif == -1)[0]
                        print('Number of spikes: ', len(spiketimes))

                        fig_v = self.show_spiketimes(voltage_trace, spiketimes, start, end, fs)

                        display(fig_v)

                        adjust1 = bool(int(input('Adjust threshold again? (Yes: 1, No: 0): ')))
                        plt.close(fig_v)

            # insert
            self.insert1(dict(key, spiketimes=spiketimes, rec_len=len(voltage_trace),nspikes=len(spiketimes)))

    def show_spiketimes(self, voltage_trace, spiketimes, start, end, fs):

        """
            :param voltage_trace array (1, rec_len) with the filtered raw trace
            :param spiketimes array (1,nSpikes) with spike times in sample points
            :param start scalar start of plottted segment in s
            :param end scalar end of plottted segment in s
            :param fs scalar sampling rate in Hz
            """

        plt.rcParams.update(
            {'figure.figsize': (15, 6), 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16,
             'ytick.labelsize': 16})

        fig, ax = plt.subplots()

        x = np.linspace(start, end, (end - start) * fs)
        n = spiketimes[(spiketimes > start * fs) & (spiketimes < end * fs)].astype(int)

        ax.plot(x, voltage_trace[start * fs:end * fs], linewidth=2)
        ax.plot(x[n - start * fs], voltage_trace[n], 'or')
        ax.set_xlim([start, end])
        ax.set_ylabel('Voltage [mV]', labelpad=20)
        ax.set_xlabel('Time [s]', labelpad=20)
        plt.locator_params(axis='y', nbins=5)

        return fig

    def plt_rawtrace(self):

        start = int(input('Plot voltage trace from (in s): '))
        end = int(input('to (in s): '))

        for key in self.project().fetch.as_dict:

            fname = key['filename']

            fs = (Recording() & key).fetch1['fs']
            ch_voltage, ch_trigger = (Recording() & key).fetch1['ch_voltage', 'ch_trigger']

            cell_path = (Cell() & key).fetch1['folder']
            exp_path = (Experiment() & key).fetch1['path']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            full_path = exp_path + cell_path + fname + '.h5'
            f = h5py.File(full_path, 'r')
            ch_grp = f[ch_voltage]
            keylist = [key for key in ch_grp.keys()]  # get key within one group
            voltage_trace = ch_grp[keylist[1]]['data'][:]  # initialize as section_00

            for sec in range(2, len(keylist)):
                ch_sec_tmp = ch_grp[keylist[sec]]
                dset = ch_sec_tmp['data'][:]  # get array
                voltage_trace = np.append(voltage_trace, dset)

            plt.rcParams.update(
                {'figure.figsize': (15, 6), 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16,
                 'ytick.labelsize': 16})

            x = np.linspace(start, end, (end - start) * fs)

            fig, ax = plt.subplots()

            plt.suptitle(str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            ax.plot(x, voltage_trace[start * fs:end * fs], linewidth=2)
            ax.set_ylabel('Voltage [mV]', labelpad=20)
            ax.set_xlabel('Time [s]', labelpad=20)
            ax.set_xlim([start, end])
            plt.locator_params(axis='y', nbins=5)

    def plt_spiketimes(self):

        start = int(input('Plot voltage trace from (in s): '))
        end = int(input('to (in s): '))

        for key in self.project().fetch.as_dict:

            fname = key['filename']

            spiketimes = (self & key).fetch1['spiketimes']

            fs = (Recording() & key).fetch1['fs']
            ch_voltage, ch_trigger = (Recording() & key).fetch1['ch_voltage', 'ch_trigger']

            cell_path = (Cell() & key).fetch1['folder']
            exp_path = (Experiment() & key).fetch1['path']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            full_path = exp_path + cell_path + fname + '.h5'

            # read rawdata from file

            f = h5py.File(full_path, 'r')
            ch_grp = f[ch_voltage]
            keylist = [key for key in ch_grp.keys()]
            voltage_trace = ch_grp[keylist[1]]['data'][:]

            for sec in range(2, len(keylist)):
                ch_sec_tmp = ch_grp[keylist[sec]]
                dset = ch_sec_tmp['data'][:]
                voltage_trace = np.append(voltage_trace, dset)

            # filter voltage trace

            wn0 = 45 / fs
            wn1 = 55 / fs
            b, a = scignal.butter(2, [wn0, wn1], 'bandstop',
                                  analog=False)  # second order, critical frequency, type, analog or digital
            voltage_trace = scignal.filtfilt(b, a, voltage_trace)

            # plot

            plt.rcParams.update(
                {'figure.figsize': (15, 6), 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16,
                 'ytick.labelsize': 16})

            fig, ax = plt.subplots()
            plt.suptitle(str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            x = np.linspace(start, end, (end - start) * fs)
            n = spiketimes[(spiketimes > start * fs) & (spiketimes < end * fs)].astype(int)

            ax.plot(x, voltage_trace[start * fs:end * fs], linewidth=2)
            ax.plot(x[n - start * fs], voltage_trace[n], 'or')
            ax.set_xlim([start, end])
            ax.set_ylabel('Voltage [mV]', labelpad=20)
            ax.set_xlabel('Time [s]', labelpad=20)
            plt.locator_params(axis='y', nbins=5)

@schema
class Trigger(dj.Computed):
    definition="""
    ->Recording
    ---
    triggertimes	:longblob	# trigger times in sample points
    """

    def _make_tuples(self,key):
        # fetch required data

        fname = (Recording() & key).fetch1['filename']
        ch_trigger = (Recording() & key).fetch1['ch_trigger']

        cell_path = (Cell() & key).fetch1['folder']
        exp_path = (Experiment() & key).fetch1['path']

        # read rawdata from file

        full_path = exp_path + cell_path + fname + '.h5'
        f = h5py.File(full_path,'r')

        ch_grp = f[ch_trigger]
        keylist = [key for key in ch_grp.keys()]
        trigger_trace  = ch_grp[keylist[1]]['data'][:]
        for sec in range(2,len(keylist)):
            ch_sec_tmp = ch_grp[keylist[sec]]
            dset = ch_sec_tmp['data'][:]
            trigger_trace = np.append(trigger_trace,dset)

        # get trigger times by diff

        tmp = np.array(trigger_trace)
        thr_boolean = [tmp < 1]
        tmp[thr_boolean] = 0
        tmp[tmp!=0]=1
        tmp2 = np.append(tmp[1:len(tmp)],[0])
        dif = tmp-tmp2
        triggertimes = np.where(dif==-1)[0]

        # insert
        self.insert1(dict(key, triggertimes=triggertimes))

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
                elif '2400' in fname:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_15_15_2400'))
                else:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_official'))


            elif ('20um' in fname) or ('BCNoise' in fname):
                if 'long' in fname:
                    self.insert1(dict(key,freq=5,delx=20, dely=20,mseq='BWNoise_long'))
                elif '2400' in fname:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_15_15_2400'))
                else:
                    self.insert1(dict(key,freq=5,delx=20, dely=20,mseq='BWNoise_official'))
            else:
                if 'long' in fname:
                    self.insert1(dict(key,freq=5,delx=40, dely=40,mseq='BWNoise_long'))
                elif '2400' in fname:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_15_15_2400'))
                else:
                    self.insert1(dict(key,freq=5,delx=40, dely=40,mseq='BWNoise_official'))

        elif '20Hz' in fname:
            if '40um' in fname:
                if 'long' in fname:
                    self.insert1(dict(key,freq=20,delx=40,dely=40,mseq='BWNoise_long'))
                elif '2400' in fname:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_15_15_2400'))
                else:
                    self.insert1(dict(key,freq=20,delx=40,dely=40,mseq='BWNoise_official'))


            elif ('20um' in fname) or ('BCNoise' in fname):
                if 'long' in fname:
                    self.insert1(dict(key,freq=20,delx=20, dely=20,mseq='BWNoise_long'))
                elif '2400' in fname:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_15_15_2400'))
                else:
                    self.insert1(dict(key,freq=20,delx=20, dely=20,mseq='BWNoise_official'))
            else:
                if 'long' in fname:
                    self.insert1(dict(key,freq=20,delx=40, dely=40,mseq='BWNoise_long'))
                elif '2400' in fname:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_15_15_2400'))
                else:
                    self.insert1(dict(key,freq=20,delx=40, dely=40,mseq='BWNoise_official'))

        else:
            if '40um' in fname:
                if 'long' in fname:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_long'))
                elif '2400' in fname:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_15_15_2400'))
                else:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_official'))


            elif ('20um' in fname) or ('BCNoise' in fname):
                if 'long' in fname:
                    self.insert1(dict(key,freq=5,delx=20, dely=20,mseq='BWNoise_long'))
                elif '2400' in fname:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_15_15_2400'))
                else:
                    self.insert1(dict(key,freq=5,delx=20, dely=20,mseq='BWNoise_official'))
            else:
                if 'long' in fname:
                    self.insert1(dict(key,freq=5,delx=40, dely=40,mseq='BWNoise_long'))
                elif '2400' in fname:
                    self.insert1(dict(key,freq=5,delx=40,dely=40,mseq='BWNoise_15_15_2400'))
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
    # Calculate the spike-triggered ensemble from noise recording
    -> Recording
    ---
    sta    : longblob	# spike-triggered average
    """
    @property
    def populated_from(self):
        return Recording() & dict(stim_type='bw_noise')

    def _make_tuples(self,key):

        fs = (Recording() & key).fetch1['fs']

        spiketimes = (Spikes() & key).fetch1['spiketimes']
        triggertimes = (Trigger() & key).fetch1['triggertimes']
        rec_len = (Spikes() & key).fetch1['rec_len']

        frames = (BWNoiseFrames() & key).fetch1['frames']
        (stim_length, stim_dim_x,stim_dim_y) = (BWNoiseFrames() & key).fetch1['stim_length','stim_dim_x','stim_dim_y']
        stim_freq = (BWNoise() & key).fetch1['freq']




        stimInd = np.zeros(rec_len).astype(int)-1

        if len(triggertimes) != stim_length:
            print('Something went wrong with the trigger detection!')

        else:

            for n in range(len(triggertimes)-1):
                stimInd[triggertimes[n]:triggertimes[n+1]-1] += int(n+1)
            stimInd[triggertimes[len(triggertimes)-1]:triggertimes[len(triggertimes)-1]+(fs/stim_freq)-1] += int(len(triggertimes))


        deltat = 1000
        delta = int(deltat/.1)
        spiketimes = spiketimes[spiketimes > triggertimes[0]+delta]
        spiketimes = spiketimes[spiketimes < triggertimes[len(triggertimes)-1] + int(fs/stim_freq)-1]
        nspikes = len(spiketimes)
        k = 100

        ste = np.zeros([nspikes,(delta+1000)/k,stim_dim_x*stim_dim_y])
        for st in range(nspikes):
            for t in range(-1000,delta,k):
                ste[st,int((t+1000)/k),:] = np.array(frames[stimInd[spiketimes[st]-t]])

        sta = np.mean(ste,0)

        self.insert1(dict(key,sta=sta))

    def plt_rf(self):

        plt.rcParams.update(
            {'figure.subplot.hspace': .2, 'figure.subplot.wspace': .3, 'figure.figsize': (15, 8), 'axes.titlesize': 16})

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            sta = (self & key).fetch1['sta']

            stimDim = (BWNoiseFrames() & key).fetch1['stim_dim_x', 'stim_dim_y']

            sta_smooth = scimage.filters.gaussian_filter(sta.reshape(sta.shape[0], stimDim[0], stimDim[1]),
                                                         [0.2, .7, .7])  # reshape and smooth with a gaussian filter
            sta_norm = sta_smooth / np.std(sta_smooth, 0)

            fig, axarr = plt.subplots(2, int(np.ceil(sta.shape[0] / 20)))
            fig.subplots_adjust(hspace=.1, wspace=.1)

            if (int(np.ceil(sta.shape[0])) % 20 == 0):
                ax = axarr.reshape(int(np.ceil(sta.shape[0] / 10)))
            else:
                ax = axarr.reshape(int(np.ceil(sta.shape[0] / 10)) + 1)
                im = ax[int(np.ceil(sta.shape[0] / 10))].imshow(np.zeros([20, 15]), cmap=plt.cm.Greys_r,
                                                                interpolation='none', clim=(-1, 1))
                ax[int(np.ceil(sta.shape[0] / 10))].set_xticks([])
                ax[int(np.ceil(sta.shape[0] / 10))].set_yticks([])

            tmp = 1

            with sns.axes_style(style='whitegrid'):

                for delt in range(0, sta.shape[0], 10):
                    im = ax[delt / 10].imshow(sta_norm[delt, :, :],
                                              cmap=plt.cm.coolwarm,
                                              clim=(-np.percentile(sta_norm, 90), np.percentile(sta_norm, 90)),
                                              interpolation='none')
                    ax[delt / 10].set_title('$\Delta$ t = ' + str(-(delt - 10) * 10) + 'ms')
                    ax[delt / 10].set_yticks([])
                    ax[delt / 10].set_xticks([])
                    tmp += 1

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label('s.d. units', labelpad=40, rotation=270)

                plt.suptitle('STA for different time lags\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

    def plt_contour(self, tau, x1, x2, y1, y2):

        from matplotlib import ticker

        plt.rcParams.update({
            'figure.figsize': (10, 8), 'figure.subplot.hspace': .2, 'figure.subplot.wspace': .2, 'axes.titlesize': 16,
            'axes.labelsize': 18,
            'xtick.labelsize': 16, 'ytick.labelsize': 16, 'lines.linewidth': 4})

        for key in self.project().fetch.as_dict:
            sta = (self & key).fetch1['sta']
            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            stimDim = (BWNoiseFrames() & key).fetch1['stim_dim_x', 'stim_dim_y']
            sta_smooth = scimage.filters.gaussian_filter(sta.reshape(sta.shape[0], stimDim[0], stimDim[1]),
                                                         [0.2, .7, .7])  # reshape and smooth with a gaussian filter

            frame = int(10 - tau / 10)

            fig = plt.figure()
            plt.title('$\Delta$ t: ' + str(tau) + '\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            im = plt.imshow(sta_smooth[frame, :, :][x1:x2, y1:y2], interpolation='none',
                            cmap=plt.cm.Greys_r, extent=(y1, y2, x2, x1), origin='upper')
            cs = plt.contour(sta_smooth[frame, :, :][x1:x2, y1:y2],
                             extent=(y1, y2, x2, x1), cmap=plt.cm.coolwarm, origin='upper', linewidth=4)

            cb = plt.colorbar(cs, extend='both', shrink=.8)
            cbaxes = fig.add_axes([.15, .02, .6, .03])  # [left, bottom, width, height]
            cbi = plt.colorbar(im, orientation='horizontal', cax=cbaxes)

            tick_locator = ticker.MaxNLocator(nbins=6)
            cbi.locator = tick_locator
            cbi.update_ticks()

            cb.locator = tick_locator
            cb.update_ticks()


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
        triggertimes = (Trigger() & key).fetch1['triggertimes']
        ntrials = int(np.floor(len(triggertimes)/2))
        self.insert1(dict(key,ntrials=ntrials))

@schema
class Chirp(dj.Computed):
    definition="""
    -> Recording
    -> ChirpParams
    ---
    psth_trials		: longblob		# psth per trial
    psth			: longblob		# average psth
    loop_duration_s	: double		# real duration of one stimulus loop
    qi_chirp		: double		# quality response index
    """

    @property
    def populated_from(self):
        return (Recording() & dict(stim_type='chirp'))


    def _make_tuples(self,key):
            triggertimes = (Trigger() & key).fetch1['triggertimes']
            spiketimes = (Spikes() & key).fetch1['spiketimes']
            ntrials = (ChirpParams() & key).fetch1['ntrials']
            fs = (Recording() & key).fetch1['fs']

            triggertimes = triggertimes.reshape(ntrials, 2)

            StimDuration = 32.5

            true_loop_duration = []
            for trial in range(1, ntrials):
                true_loop_duration.append(triggertimes[trial, 0] - triggertimes[trial - 1, 0])

            loop_duration_n = np.ceil(np.mean(true_loop_duration))  # in sample points
            loop_duration_s = loop_duration_n / fs  # in s

            print('Due to imprecise stimulation freqeuncy a delta of', loop_duration_s - StimDuration,
                  's was detected')

            f = []
            for trial in range(ntrials - 1):
                f.append(np.array(spiketimes[(spiketimes > triggertimes[trial, 0]) & (spiketimes < triggertimes[trial + 1, 0])]))
            f.append(np.array(
                spiketimes[(spiketimes > triggertimes[ntrials - 1, 0]) & (spiketimes < triggertimes[ntrials - 1, 0] + loop_duration_n)]))

            f_norm = []
            for trial in range(ntrials):
                f_norm.append(f[trial] - triggertimes[trial, 0])


            T = int(loop_duration_s)  # in s
            delT = .1

            nbins1 = T / delT

            psth = np.zeros(nbins1)  # .astype(int)
            psth_trials = []

            for trial in range(ntrials):
                psth_trials.append(np.histogram(f_norm[trial] / fs, nbins1, [0, T])[0])
                psth += psth_trials[trial]

                psth = psth / (delT * ntrials)

            R = np.array(psth_trials).transpose()
            qi_chirp = np.var(np.mean(R,1))/np.mean(np.var(R,0)).astype(float)

            self.insert1(dict(key,psth_trials = np.array(psth_trials),psth= np.array(psth),loop_duration_s = loop_duration_s,qi_chirp = qi_chirp))


    def plt_chirp(self):

        # Plotting parameter
        plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.labelsize': 16, 'axes.titlesize': 20,
                             'figure.figsize': (12, 8), 'lines.linewidth': 2})

        # define stimulus


        for key in self.project().fetch.as_dict:

            loop_duration_s = (self & key).fetch1['loop_duration_s']
            psth = (self & key).fetch1['psth']
            fname = (self & key).fetch1['filename']

            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            fs = (Recording() & key).fetch1['fs']

            spiketimes = (Spikes() & key).fetch1['spiketimes']
            triggertimes = (Trigger() & key).fetch1['triggertimes']

            ntrials = int((ChirpParams() & key).fetch1['ntrials'])
            triggertimes = triggertimes.reshape(ntrials,2)

            delT = .1
            loop_duration_n = loop_duration_s * fs

            f = []
            for trial in range(ntrials - 1):
                f.append(np.array(spiketimes[(spiketimes > triggertimes[trial, 0]) & (spiketimes < triggertimes[trial + 1, 0])]))
            f.append(np.array(
                spiketimes[(spiketimes > triggertimes[ntrials - 1, 0]) & (spiketimes < triggertimes[ntrials - 1, 0] + loop_duration_n)]))

            f_norm = []
            for trial in range(ntrials):
                f_norm.append(f[trial] - triggertimes[trial, 0])


            ChirpDuration = 8                   # Time (s) of rising/falling chirp phase
            ChirpMaxFreq  = 8                   # Peak frequency of chirp (Hz)
            IntensityFrequency = 2               # freq at which intensity is modulated


            SteadyOFF     = 3.00                 # Time (s) of Light OFF at beginning at end of stimulus
            SteadyOFF2     = 2.00
            SteadyON      = 3.00                 # Time (s) of Light 100% ON before and after chirp
            SteadyMID     = 2.00                 # Time (s) of Light at 50% for steps

            Fduration     = 0.017               # Single Frame duration (s) -  ADJUST DEPENDING ON MONITOR


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


            T = int(loop_duration_s) # in s
            delT = .1 # in s

            nbins1 = T/delT

            tPSTH = np.linspace(0,T,nbins1)

            fig, axarr = plt.subplots(3, 1, sharex=True)
            plt.subplots_adjust(hspace=.7)
            plt.suptitle('Chirp\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            for trial in range(ntrials):
                axarr[1].scatter(f_norm[trial] / 10000, trial * np.ones([len(f_norm[trial] / 10000)]),
                                 color='k')  # scatter(tStar{trial},trial*ones(1,length(tStar{trial})),'b.')
                axarr[1].set_ylabel('# trial', labelpad=20)

            axarr[1].set_ylim(-.5, ntrials - .5)
            axarr[1].set_yticklabels(np.linspace(0, ntrials, ntrials + 1).astype(int))

            axarr[2].plot(tPSTH, psth, 'k')
            axarr[2].set_ylabel('PSTH', labelpad=10)
            axarr[2].set_yticks([0, max(psth) / 2, max(psth)])
            axarr[2].set_xlabel('time [s]')

            (tChirp, chirp) = stimulus()

            axarr[0].plot(tChirp, chirp, 'k')
            axarr[0].set_ylabel('stimulus intensity', labelpad=5)
            axarr[0].set_yticks([0, 127, 250])
            axarr[0].set_xlim(0, loop_duration_s)


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

        spiketimes = (Spikes() & key).fetch1['spiketimes']
        triggertimes = (Trigger() & key).fetch1['triggertimes']
        nconditions = (DSParams() & key).fetch1['nconditions']

        ntrials = int(len(triggertimes)/nconditions)

        deg = np.array([0,180,45,225,90,270,135,315]) #np.arange(0, 360, 360/nconditions)
        idx = np.array([0,4,6,2,5,1,7,3])

        true_loop_duration = []
        for trial in range(1,ntrials):
            true_loop_duration.append(triggertimes[trial*nconditions] - triggertimes[(trial-1)*nconditions])
        loop_duration_n =  np.ceil(np.mean(true_loop_duration)) # in sample points
        #loopDuration_s = loopDuration_n/10000 # in s


        spikes_trial = []
        spikes_normed = []
        hist = []
        hist_sorted = []

        for trial in range(ntrials-1):
            spikes_trial.append(np.array(spiketimes[(spiketimes>triggertimes[trial*nconditions]) & (spiketimes<triggertimes[(trial+1)*nconditions])]))
            spikes_normed.append(spikes_trial[trial]-triggertimes[trial*nconditions])
            hist.append(np.histogram(spikes_normed[trial],8,[0,loop_duration_n])[0])

            # sort by condition
            hist_sorted.append(hist[trial][idx])

        spikes_trial.append(np.array(spiketimes[(spiketimes > triggertimes[(ntrials-1)*nconditions])
                                            & (spiketimes < triggertimes[(ntrials-1)*nconditions]+loop_duration_n)]))
        spikes_normed.append(spikes_trial[ntrials-1]-triggertimes[(ntrials-1)*nconditions])
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

        plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.labelsize': 16, 'axes.titlesize': 20,
                             'figure.figsize': (10, 8)})

        for key in self.project().fetch.as_dict:
            hist = (self & key).fetch1['hist']
            nconditions = (DSParams() & key).fetch1['nconditions']
            fname = key['filename']
            qi = (self & key).fetch1['qi']
            dsi = (self & key).fetch1['dsi']

            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            deg = np.array([0, 180, 45, 225, 90, 270, 135, 315])

            with sns.axes_style('whitegrid'):
                fig = plt.figure()
                plt.suptitle('Directional Tuning\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=18, y=1.1)

                ax = plt.axes(polar=True, axisbg='white')
                width = .2
                rads = np.radians(deg) - width / 2
                counts = np.mean(hist, 0)
                plt.bar(rads, counts, width=width, facecolor='k')

                ycounts = [round(max(counts) / 2), round(max(counts))]
                ax.set_theta_direction(-1)
                ax.set_theta_offset(np.pi / 2)
                ax.set_yticks(ycounts)
                ax.grid(color='k', linestyle='--')

                ax.annotate('QI: ' + str("%.2f" % round(qi, 2)), xy=(.85, .2), xycoords='figure fraction', size=16)
                ax.annotate('DSI: ' + str("%.2f" % round(dsi, 2)), xy=(.9, .3), xycoords='figure fraction', size=16)
                # ax.annotate('.85,.2', xy = (.85,.2), xycoords = 'figure fraction')
                # ax.annotate('.9,.3', xy = (.9,.3), xycoords = 'figure fraction')

    def plt_ds_traces(self):

        plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.labelsize': 16, 'axes.titlesize': 20,
                             'figure.figsize': (15, 8)})

        for key in self.project().fetch.as_dict:

            fname = (Recording() & key).fetch1['filename']
            ch_trigger = (Recording() & key).fetch1['ch_trigger']
            ch_voltage = (Recording() & key).fetch1['ch_voltage']
            rec_len = (Spikes() & key).fetch1['rec_len']

            cell_path = (Cell() & key).fetch1['folder']
            exp_path = (Experiment() & key).fetch1['path']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            # extract raw data for the given recording
            full_path = exp_path + cell_path + fname + '.h5'
            f = h5py.File(full_path, 'r')

            ch_grp_v = f[ch_voltage]
            keylist = [key for key in ch_grp_v.keys()]
            voltage_trace = ch_grp_v[keylist[1]]['data'][:]
            for sec in range(2, len(keylist)):
                ch_sec_tmp = ch_grp_v[keylist[sec]]
                dset = ch_sec_tmp['data'][:]
                voltage_trace = np.append(voltage_trace, dset)

            triggertimes = (Trigger() & key).fetch1['triggertimes']

            # stimulus parameter
            fs = (Recording() & key).fetch1['fs']
            t_off = 2 * fs  # in s * fs
            t_on = 2 * fs  # in s

            nconditions = (DSParams() & key).fetch1['nconditions']
            ntrials = int(len(triggertimes) / nconditions)
            idx = np.array([0, 4, 6, 2, 5, 1, 7, 3])
            deg = np.arange(0, 360, 360 / 8).astype(int)

            true_loop_duration = []
            for trial in range(1, ntrials):
                true_loop_duration.append(triggertimes[trial * nconditions] - triggertimes[(trial - 1) * nconditions])
            loop_duration_n = np.ceil(np.mean(true_loop_duration))

            stim = np.zeros(rec_len)

            for i in range(len(triggertimes)):
                stim[triggertimes[i]:triggertimes[i] + t_on] = 1

            v_trace_trial = []
            stim_trial = []
            for i in range(len(triggertimes)):
                v_trace_trial.append(np.array(voltage_trace[triggertimes[i]:triggertimes[i] + t_on + t_off]))
                stim_trial.append(np.array(stim[triggertimes[i]:triggertimes[i] + t_on + t_off]))

            plt.rcParams.update({'figure.subplot.hspace': .1, 'figure.figsize': (20, 8)})

            N = len(v_trace_trial)
            fig1, axarr = plt.subplots(int(N / nconditions) + 1, nconditions, sharex=True,
                                       sharey=True)  # len(triggertimes)

            for i in range(N + nconditions):
                rowidx = int(np.floor(i / nconditions))
                colidx = int(i - rowidx * nconditions)
                if rowidx == 0:
                    axarr[rowidx, colidx].plot(stim_trial[i] * np.max(v_trace_trial) - np.max(v_trace_trial), 'k')
                    axarr[rowidx, colidx].set_xticks([])
                    axarr[rowidx, colidx].set_yticks([])
                else:
                    axarr[rowidx, colidx].plot(v_trace_trial[i - nconditions], 'k')
                    axarr[rowidx, colidx].set_xticks([])
                    axarr[rowidx, colidx].set_yticks([])
            plt.suptitle(
                'Traces sorted by direction (column) and trial (row)\n' + str(exp_date) + ': ' + eye + ': ' + fname,
                fontsize=16)

            rec_type = (Recording() & key).fetch1['rec_type']

            # Heatmap
            if rec_type == 'intracell':
                arr = np.array(v_trace_trial)
                arr = arr.reshape(ntrials, nconditions, arr.shape[1])

                for trial in range(ntrials):
                    arr[trial, :, :] = arr[trial, :, :][idx]

                l = []
                for cond in range(nconditions):
                    for trial in range(ntrials):
                        l.append(arr[trial, cond, :])

                fig2, ax = plt.subplots()

                intensity = np.array(l).reshape(ntrials, nconditions, len(l[0]))

                column_labels = np.linspace(0, 4, 5)
                row_labels = deg.astype(int)

                plt.pcolormesh(np.mean(intensity, 0), cmap=plt.cm.coolwarm)
                cax = plt.colorbar()
                cax.set_label('voltage [mV]', rotation=270, labelpad=50)
                plt.xlabel('time [s]')
                plt.ylabel('direction [deg]')
                plt.title('Average membrane potential')

                ax.set_xticks(np.linspace(0, len(l[0]), 5), minor=False)
                ax.set_yticks(np.arange(intensity.shape[1]) + .5, minor=False)

                ax.invert_yaxis()
                ax.xaxis.set_ticks_position('bottom')

                ax.set_xticklabels(column_labels, minor=False)
                ax.set_yticklabels(row_labels, minor=False)


@schema
class OnOff(dj.Computed):
    definition="""
    -> Recording
    ---
    qi  :double # quality response index
    pol :double # on off polarit index
    """

    @property
    def populated_from(self):

        return Recording() & dict(stim_type='on_off')

    def _make_tuples(self, key):

        # fetch data

        spiketimes = (Spikes() & key).fetch1['spiketimes']
        triggertimes = (Trigger() & key).fetch1['triggertimes']
        fs = (Recording() & key).fetch1['fs']

        t_off = .5 * fs  # in s * fs
        t_on = .5 * fs  # in s * fs (samplepoints)

        spiketimes = spiketimes[spiketimes > triggertimes[0]]
        spiketimes = spiketimes[spiketimes < triggertimes[len(triggertimes) - 1] + 2 * t_off + 2 * t_on]

        spikes_on = []
        spikes_off = []
        spikes_norm = []

        for tr in triggertimes:
            spikes_norm.append(spiketimes[spiketimes > tr & spiketimes <  tr + 2*t_off + 2*t_on])
            spikes_off.append(spiketimes[(spiketimes > tr) & (spiketimes < tr + t_off)])
            spikes_on.append(spiketimes[(spiketimes > tr + t_off) & (spiketimes < tr + t_off + 2 * t_on)])


        pol = len(spikes_on) / len(spikes_off)
        self.insert1(dict(key, pol=pol))

    def plt_on_off(self):

        for key in self.project().fetch.as_dict:

            fname = (Recording() & key).fetch1['filename']
            ch_trigger = (Recording() & key).fetch1['ch_trigger']
            ch_voltage = (Recording() & key).fetch1['ch_voltage']
            rec_type = (Recording() & key).fetch1['rec_type']
            fs = (Recording() & key).fetch1['fs']

            cell_path = (Cell() & key).fetch1['folder']
            exp_path = (Experiment() & key).fetch1['path']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            # extract raw data for the given recording
            full_path = exp_path + cell_path + fname + '.h5'
            f = h5py.File(full_path, 'r')

            ch_grp_v = f[ch_voltage]
            keylist = [key for key in ch_grp_v.keys()]
            voltage_trace = ch_grp_v[keylist[1]]['data'][:]
            for sec in range(2, len(keylist)):
                ch_sec_tmp = ch_grp_v[keylist[sec]]
                dset = ch_sec_tmp['data'][:]
                voltage_trace = np.append(voltage_trace, dset)

            triggertimes = (Trigger() & key).fetch1['triggertimes']
            offset_stim = .027  # in s
            triggertimes = triggertimes + offset_stim * fs

            plt.rcParams.update(
                {'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.labelsize': 16, 'axes.titlesize': 20,
                 'figure.figsize': (10, 8), 'lines.linewidth': 2, 'figure.subplot.hspace': .2,
                 'figure.subplot.hspace': .2})

            t_off = .5 * fs  # in s * fs
            t_on = .5 * fs  # in s

            stim = np.zeros(voltage_trace.shape)

            for i in range(len(triggertimes)):
                stim[triggertimes[i] + t_off:triggertimes[i] + t_off + 2 * t_on] = 1

            v_trace_trial = []
            stim_trial = []
            for i in range(len(triggertimes) - 1):
                v_trace_trial.append(np.array(voltage_trace[triggertimes[i]:triggertimes[i] + 2 * t_off + 2 * t_on]))
                stim_trial.append(np.array(stim[triggertimes[i]:triggertimes[i] + 2 * t_off + 2 * t_on]))

            scale = np.max(voltage_trace) - np.min(voltage_trace)
            offset = np.min(voltage_trace)

            plt.rcParams.update(
                {'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
                 'figure.figsize': (15, 8), 'figure.subplot.hspace': .1})

            fig1, axarr = plt.subplots(4, int(np.ceil(len(triggertimes) / 2)), sharex=True, sharey=True)
            plt.suptitle('Spot response\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=18)

            for i in range(len(v_trace_trial)):
                rowidx = 2 * int(np.ceil((i + 1) / (len(v_trace_trial) * .5)) - 1)
                colidx = int(i - (rowidx) * len(v_trace_trial) * .5)
                axarr[rowidx, colidx].plot(stim_trial[i] * scale + offset, 'k', linewidth=2)
                axarr[rowidx, colidx].set_xticks([])
                axarr[rowidx, colidx].set_yticks([])
                axarr[rowidx + 1, colidx].plot(v_trace_trial[i], 'k')
                axarr[rowidx + 1, colidx].set_xticks([])
                axarr[rowidx + 1, colidx].set_yticks([])

            # Plot heatmap

            if rec_type == 'intracell':
                fig2, ax = plt.subplots()

                intensity = np.array(v_trace_trial)

                plt.pcolormesh(intensity, cmap=plt.cm.coolwarm)

                cax = plt.colorbar()
                cax.set_label('voltage [mV]', rotation=270, labelpad=50)

                ax.set_ylim([0, len(v_trace_trial)])
                ax.set_xticks(np.linspace(0, intensity.shape[1], 5), minor=False)
                # ax.set_yticks(np.linspace(0,intensity.shape[0],5), minor=False)

                ax.invert_yaxis()
                ax.xaxis.set_ticks_position('bottom')

                # row_labels = np.linspace(0,intensity.shape[0],5)
                column_labels = np.linspace(0, 2, 5)
                ax.set_xticklabels(column_labels, minor=False)
                # ax.set_yticklabels(row_labels, minor=False)

                plt.xlabel('time [s]')
                plt.ylabel('trial')
                plt.title('Membrane potential')


def addEntry(animal_id,sex,date_of_birth,exp_date,eye,cell_id,morph,cell_type,data_folder,filename,rec_type,ch_voltage):
    """

    :param animal_id: str 'ZK0-yyyy-mm-dd'
    :param sex: str 'F' or 'M'
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
    #from schema import Animal,Experiment,Cell,Recording
    A = Animal()
    E = Experiment()
    C = Cell()
    R = Recording()
    try:
        A.insert1({'animal_id':animal_id,'sex':sex,'date_of_birth':date_of_birth})
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
