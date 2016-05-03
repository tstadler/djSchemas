import datajoint as dj
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as scimage
import scipy.signal as scignal

schema = dj.schema('ageuler_rgcEphys_test2',locals())

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
    experimenter="tstadler"		:varchar(20)				            # first letter of first name + last name = lrogerson/tstadler
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



    def show_spiketimes(self,voltage_trace,spiketimes, start, end, fs):

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

            plt.rcParams.update(
            {'figure.figsize': (15, 6), 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16,
             'ytick.labelsize': 16})

            x = np.linspace(start, end, (end - start) * fs)

            fig, ax = plt.subplots()

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
            ch_voltage, ch_trigger = (Recording() & key).fetch1['ch_voltage','ch_trigger']

            cell_path = (Cell() & key).fetch1['folder']
            exp_path = (Experiment() & key).fetch1['path']

            full_path = exp_path + cell_path + fname + '.h5'

            # read rawdata from file

            f = h5py.File(full_path,'r')
            ch_grp = f[ch_voltage]
            keylist = [key for key in ch_grp.keys()]
            voltage_trace  = ch_grp[keylist[1]]['data'][:]

            for sec in range(2,len(keylist)):
                ch_sec_tmp = ch_grp[keylist[sec]]
                dset = ch_sec_tmp['data'][:]
                voltage_trace = np.append(voltage_trace,dset)

            # filter voltage trace

            wn0 = 45 / fs
            wn1 = 55 / fs
            b, a = scignal.butter(2, [wn0, wn1], 'bandstop',
                                  analog=False)  # second order, critical frequency, type, analog or digital
            voltage_trace = scignal.filtfilt(b, a, voltage_trace)

            # plot

            plt.rcParams.update({'figure.figsize': (15, 6), 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16,
             'ytick.labelsize': 16})

            x = np.linspace(start, end, (end - start) * fs)

            fig, ax = plt.subplots()

            ax.plot(x, voltage_trace[start * fs:end * fs], linewidth=2)
            ax.set_ylabel('Voltage [mV]', labelpad=20)
            ax.set_xlabel('Time [s]', labelpad=20)
            ax.set_xlim([start, end])
            plt.locator_params(axis='y', nbins=5)