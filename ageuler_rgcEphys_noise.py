import datajoint as dj
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as scimage
import scipy.signal as scignal
import scipy.misc as scmisc
from IPython.display import display
from matplotlib import ticker
import matplotlib.gridspec as gridsp
import tifffile as tf
from sklearn.preprocessing import binarize
from configparser import ConfigParser
import scipy.optimize as scoptimize
import scipy.stats as scstats
from sklearn.cross_validation import KFold
import pandas as pd

schema = dj.schema('ageuler_rgcEphys_noise',locals())

@schema
class Animal(dj.Manual):
    definition = """
    # Basic animal info

    animal_id					:varchar(50)   																# unique ID given to the animal
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

    folder		:varchar(200)		#relative folder path for the subexperiment
    morphology  : boolean           # morphology was recorded or not
    type        : varchar(200)      # putative RGC type
    """

@schema
class Recording(dj.Manual):
    definition = """
    # Information for a particular recording file

    -> Cell
    filename		: varchar(200) 		  						# name of the converted recording file
    ---
    rec_type                            : enum('intracell','extracell')                     # recording is an intra- or juxtacellular recordin
    stim_type							: enum('bw_noise','chirp','ds','on_off','unknown')	# type of the stimulus played during recording
    fs=10000							: int					  					# sampling rate of the recording
    ch_voltage="Vm_scaled AI #10"		: varchar(50)								# name of the voltage channel for this recording
    ch_trigger="LightTrig Blanking"		: varchar(50)								# name of the channel containing the light trigger signal


    """

@schema
class Spikes(dj.Computed):
    definition = """
    # Spike times in the Recording

    -> Recording
    ---
    spiketimes		: longblob				# array spiketimes (1,nSpikes) containing spiketimes  in sample points
    nspikes         : int                   # number of spikes detected
    rec_len			: int					# number of data points sampled

    """

    def _make_tuples(self, key):

        # fetch required data

        filename = (Recording() & key).fetch1['filename']
        ch_voltage = (Recording() & key).fetch1['ch_voltage']
        rec_type = (Recording() & key).fetch1['rec_type']
        fs = (Recording() & key).fetch1['fs']

        cell_path = (Cell() & key).fetch1['folder']
        exp_path = (Experiment() & key).fetch1['path']

        full_path = exp_path + cell_path + filename + '.h5'

        # extract raw data for the given recording

        f = h5py.File(full_path, 'r')

        ch_grp = f[ch_voltage]  # get each channel group into hdf5 grp object
        keylist = [key for key in ch_grp.keys()]  # get key within one group
        voltage_trace = ch_grp[keylist[1]]['data'][:]  # initialize as section_00

        for sec in range(2, len(keylist)):
            ch_sec_tmp = ch_grp[keylist[sec]]
            dset = ch_sec_tmp['data'][:]  # get array
            voltage_trace = np.append(voltage_trace, dset)

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

        start = 30
        end = 60

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

            if rec_type == 'intracell':

                adjust1 = True

                while adjust1:
                    alpha = int(input('Scale factor for threshold: '))

                    d_voltage = np.append(np.array(np.diff(voltage_trace)), np.array(0))
                    d_sigma = np.median(np.abs(d_voltage + np.abs(min(d_voltage))) / (.6745))

                    d_tmp = np.array(d_voltage + np.abs(min(d_voltage)))

                    d_tmp[d_tmp < alpha * d_sigma] = 0
                    d_tmp[d_tmp > alpha * d_sigma] = 1

                    d_tmp = d_tmp.astype(int)
                    d_tmp2 = np.append(d_tmp[1:len(d_tmp)], np.array([0], int))

                    d_dif = d_tmp2 - d_tmp

                    spiketimes = np.where(d_dif == -1)[0]
                    print('Number of spikes: ', len(spiketimes))
                    print('Threshold for differentiated signal was: ', d_sigma * alpha)

                    fig_v = self.show_spiketimes(voltage_trace, spiketimes, start, end, fs)
                    fig_dv = self.show_spiketimes(d_voltage, spiketimes, start, end, fs)

                    display(fig_v, fig_dv)

                    adjust1 = bool(int(input('Adjust threshold again? (Yes: 1, No: 0): ')))
                    plt.close(fig_v)
                    plt.close(fig_dv)

        # insert
        self.insert1(dict(key, spiketimes=spiketimes, rec_len=len(voltage_trace), nspikes=len(spiketimes)))

    def show_spiketimes(self, voltage_trace, spiketimes, start, end, fs):

        """
            :param voltage_trace array (1, rec_len) with the filtered raw trace
            :param spiketimes array (1,nSpikes) with spike times in sample points
            :param start scalar start of plottted segment in s
            :param end scalar end of plottted segment in s
            :param fs scalar sampling rate in Hz
            """

        plt.rcParams.update(
            {'figure.figsize': (14, 7),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2,
             'lines.linewidth': 2
             }
        )

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

            plt.rcParams.update(
                {'figure.figsize': (14, 7),
                 'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'figure.subplot.hspace': .2,
                 'figure.subplot.wspace': .2,
                 'lines.linewidth': 2
                 }
            )

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

            x = np.linspace(start, end, (end - start) * fs)

            fig, ax = plt.subplots()

            plt.suptitle(str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            ax.plot(x, voltage_trace[start * fs:end * fs], linewidth=2)
            ax.set_ylabel('Voltage [mV]', labelpad=20)
            ax.set_xlabel('Time [s]', labelpad=20)
            ax.set_xlim([start, end])
            plt.locator_params(axis='y', nbins=5)

            plt.tight_layout()
            plt.subplots_adjust(top=.8)

            return fig

    def plt_spiketimes(self):

        start = int(input('Plot voltage trace from (in s): '))
        end = int(input('to (in s): '))

        for key in self.project().fetch.as_dict:

            plt.rcParams.update(
                {'figure.figsize': (14, 7),
                 'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'figure.subplot.hspace': .2,
                 'figure.subplot.wspace': .2,
                 'lines.linewidth': 2
                 }
            )

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

            plt.tight_layout()
            plt.subplots_adjust(top=.8)

            return fig

@schema
class Trigger(dj.Computed):
    definition = """
    ->Recording
    ---
    triggertimes	:longblob	# trigger times in sample points
    ntrigger        :int        # n of triggers = n of stimulus frames displayed
    """

    def _make_tuples(self, key):
        # fetch required data

        fname = (Recording() & key).fetch1['filename']
        ch_trigger = (Recording() & key).fetch1['ch_trigger']

        cell_path = (Cell() & key).fetch1['folder']
        exp_path = (Experiment() & key).fetch1['path']

        # read rawdata from file

        full_path = exp_path + cell_path + fname + '.h5'
        f = h5py.File(full_path, 'r')

        ch_grp = f[ch_trigger]
        keylist = [key for key in ch_grp.keys()]
        trigger_trace = ch_grp[keylist[1]]['data'][:]
        for sec in range(2, len(keylist)):
            ch_sec_tmp = ch_grp[keylist[sec]]
            dset = ch_sec_tmp['data'][:]
            trigger_trace = np.append(trigger_trace, dset)

        # get trigger times by diff

        tmp = np.array(trigger_trace)
        thr_boolean = [tmp < 1]
        tmp[thr_boolean] = 0
        tmp[tmp != 0] = 1
        tmp2 = np.append(tmp[1:len(tmp)], [0])
        dif = tmp - tmp2
        triggertimes = np.where(dif == -1)[0]

        # insert
        self.insert1(dict(key, triggertimes=triggertimes,ntrigger = int(len(triggertimes))))

    def plt_rawtrace(self):

        start = int(input('Plot trigger trace from (in s): '))
        end = int(input('to (in s): '))

        for key in self.project().fetch.as_dict:

            plt.rcParams.update(
                {'figure.figsize': (14, 7),
                 'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'figure.subplot.hspace': .2,
                 'figure.subplot.wspace': .2,
                 'lines.linewidth': 2
                 }
            )

            fname = key['filename']

            fs = (Recording() & key).fetch1['fs']
            ch_voltage, ch_trigger = (Recording() & key).fetch1['ch_voltage', 'ch_trigger']

            cell_path = (Cell() & key).fetch1['folder']
            exp_path = (Experiment() & key).fetch1['path']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            full_path = exp_path + cell_path + fname + '.h5'
            f = h5py.File(full_path, 'r')
            ch_grp = f[ch_trigger]
            keylist = [key for key in ch_grp.keys()]  # get key within one group
            trigger_trace = ch_grp[keylist[1]]['data'][:]  # initialize as section_00

            for sec in range(2, len(keylist)):
                ch_sec_tmp = ch_grp[keylist[sec]]
                dset = ch_sec_tmp['data'][:]  # get array
                trigger_trace = np.append(trigger_trace, dset)

            x = np.linspace(start, end, (end - start) * fs)

            fig, ax = plt.subplots()

            plt.suptitle(str(exp_date) + ': ' + eye + ': ' + fname, fontsize=18)

            ax.plot(x, trigger_trace[start * fs:end * fs], linewidth=2)
            ax.set_ylabel('Trigger [mV]', labelpad=20)
            ax.set_xlabel('Time [s]', labelpad=20)
            ax.set_xlim([start, end])
            plt.locator_params(axis='y', nbins=5)

            plt.tight_layout()
            plt.subplots_adjust(top=.8)

            return fig

    def plt_triggertimes(self):

        start = int(input('Plot trigger trace from (in s): '))
        end = int(input('to (in s): '))

        for key in self.project().fetch.as_dict:

            plt.rcParams.update(
                {'figure.figsize': (14, 7),
                 'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'figure.subplot.hspace': .2,
                 'figure.subplot.wspace': .2,
                 'lines.linewidth': 2
                 }
            )

            fname = key['filename']

            triggertimes = (self & key).fetch1['triggertimes']

            fs = (Recording() & key).fetch1['fs']
            ch_voltage, ch_trigger = (Recording() & key).fetch1['ch_voltage', 'ch_trigger']

            cell_path = (Cell() & key).fetch1['folder']
            exp_path = (Experiment() & key).fetch1['path']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            full_path = exp_path + cell_path + fname + '.h5'

            # read rawdata from file

            f = h5py.File(full_path, 'r')
            ch_grp = f[ch_trigger]
            keylist = [key for key in ch_grp.keys()]
            trigger_trace = ch_grp[keylist[1]]['data'][:]

            for sec in range(2, len(keylist)):
                ch_sec_tmp = ch_grp[keylist[sec]]
                dset = ch_sec_tmp['data'][:]
                trigger_trace = np.append(trigger_trace, dset)

            # plot

            fig, ax = plt.subplots()
            plt.suptitle(str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            x = np.linspace(start, end, (end - start) * fs)
            n = triggertimes[(triggertimes > start * fs) & (triggertimes < end * fs)].astype(int)

            ax.plot(x, trigger_trace[start * fs:end * fs], linewidth=2)
            ax.plot(x[n - start * fs], trigger_trace[n], 'or')
            ax.set_xlim([start, end])
            ax.set_ylabel('Trigger [mV]', labelpad=20)
            ax.set_xlabel('Time [s]', labelpad=20)
            plt.locator_params(axis='y', nbins=5)

            plt.tight_layout()
            plt.subplots_adjust(top=.8)

            return fig


@schema
class StimMeta(dj.Computed):
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
class Stim(dj.Computed):
    definition="""
    # Stimulus frames for a given mseq
    ->Recording
    ->StimMeta
    ---
    sc   	: longblob		# centered stimulus as a (ns x t) array, where ns = ns_x x ns_y
    t	    : int			# number of frames
    ns_x	: int			# number of pixel rows
    ns_y	: int			# number of pixel columns
    """

    @property
    def populated_from(self):
        return Recording() & dict(stim_type='bw_noise')

    def _make_tuples(self, key):

        stim_folder = "/notebooks/Notebooks/Stadler/stimulus_ana/"
        mseq = (StimMeta() & key).fetch1['mseq']

        # read stimulus information into np.array
        mseq_name = stim_folder + mseq
        frames = []
        stim_dim = []

        lines = open(mseq_name + '.txt').read().split('\n')

        params = lines[0].split(',')
        stim_dim.append(int(params[0]))
        stim_dim.append(int(params[1]))
        stim_dim.append(int(params[2]))

        ns = stim_dim[0] * stim_dim[1] # spatial dimension

        for l in range(1, len(lines)):
            split = lines[l].split(',')
            frame = np.array(split).astype(int)
            frames.append(frame)

        # Whiten the stimulus ensemble

        Frames = np.array(frames).T # stimulus as a (ns x T) array
        m0 = np.mean(Frames, 1)
        C0 = np.cov(Frames)

        # Use PCA Whitening

        Fc = (Frames - m0[:, None])  # center
        el, ev = np.linalg.eig(C0)  # eigenvalue decomposition
        Q = np.diag(1 / np.sqrt(el))
        S = np.dot(Q, np.dot(ev.T, Fc))  # whitening transform

        self.insert1(dict(key, sc = Fc, t = stim_dim[2], ns_x = int(stim_dim[0]), ns_y = int(stim_dim[1])))

@schema
class Sta(dj.Computed):
    definition = """
        # Calculate the spike-triggered ensemble from noise recording
        -> Stim
        -> Spikes
        -> Trigger
        ---
        sta         : longblob	# spike-triggered average
        kernel      : longblob  # temporal kernel from center pixel with highest s.d.
        rf          : longblob  # sta map at idt
        wt          : longblob  # first temporal filter component
        ev          : longblob  # singular values
        ws          : longblob  # first spatial filter component
        tfuture     : int       # lag before spike in ms
        tpast       : int       # lag into past for each spiketime
        ids         : int       # idx of center pixel in space coordinates
        idt         : int       # idx of time lag at which kernel reaches its absolute peak
        """

    @property
    def populated_from(self):
        return Recording() & dict(stim_type='bw_noise')

    def _make_tuples(self, key):

        fs = (Recording() & key).fetch1['fs']

        rec_len = (Spikes() & key).fetch1['rec_len']
        spiketimes = (Spikes() & key).fetch1['spiketimes']
        triggertimes = (Trigger() & key).fetch1['triggertimes']


        freq = (StimMeta() & key).fetch1['freq']

        ns_x, ns_y = (Stim() & key).fetch1['ns_x','ns_y']
        ns = ns_x * ns_y

        Sc = (Stim() & key).fetch1['sc']

        ntrigger = int(len(triggertimes))

        delta_future = 100  # time lag after spike in [ms]
        delta_past = 900  # time lag before spike in [ms] is delta_past - delta_future

        npast = int(delta_past * fs * 1e-3)
        nfuture = int(delta_future * fs * 1e-3)

        nt = 10  # determines how fine the time lag is sampled

        kn = int((npast + nfuture) / (nt))  # sampling of stimulus in 100 ms steps


        spiketimes = spiketimes[spiketimes > triggertimes[0] + npast]
        spiketimes = spiketimes[spiketimes < triggertimes[ntrigger - 1] + int(fs / freq) - 1]
        nspikes = int(len(spiketimes))
        Scut = Sc[:, 0:ntrigger]


        stimInd = np.zeros(rec_len).astype(int) - 1
        for n in range(ntrigger - 1):
            stimInd[triggertimes[n]:triggertimes[n + 1] - 1] += int(n + 1)
        stimInd[triggertimes[ntrigger - 1]:triggertimes[ntrigger - 1] + (fs / freq) - 1] += int(ntrigger)

        ste = np.zeros([ns, nt, nspikes])

        for sp in range(nspikes):
            for t in range(-nfuture, npast, kn):
                ste[:, int((t + nfuture) / kn), sp] = np.array(Scut[:, stimInd[spiketimes[sp] - t]])
        sta = ste.sum(axis=2) / nspikes
        sd_map = np.std(sta, 1)

        idc = sd_map.argmax()  # spatial filter center idx
        kernel = np.mean(sta[idc - 1:idc + 1, :], 0)
        idt = abs(kernel).argmax()
        rf = sta[:, idt]

        try:
            Ws, ev, Wt = np.linalg.svd(sta)
        except Exception as e1:
            print('SVD failed due to: \n ', e1)
            wt = np.zeros(nt)
            ws = np.zeros(ns)
            ev = 0

        wt = Wt[0, :]
        ws = Ws[:, 0]

        if np.sign(rf[idc]) != np.sign(ws[idc]):
            ws = -1 * ws

        if np.sign(kernel[idt]) != np.sign(wt[idt]):
            wt = -1 * wt

        self.insert1(dict(key,
                            sta = sta,
                            kernel = kernel,
                            rf = rf,
                            ws =  ws,
                            ev = ev,
                            wt = wt,
                            tfuture = delta_future,
                            tpast = delta_past,
                            ids = idc,
                            idt = idt,
        ))


    def plt_deltas(self):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            (ns_x,ns_y) = (Stim() & key).fetch1['ns_x','ns_y']
            sta = (self & key).fetch1['sta']
            ns,nt = sta.shape
            delta_future,delta_past = (self & key).fetch1['tfuture','tpast']
            kt = (delta_past + delta_future) / nt


            fig, axarr = plt.subplots(2, int(nt / 2))
            ax = axarr.flatten()
            clim = (sta.min(), sta.max())

            for tau in range(int(len(ax))):
                im = ax[tau].imshow(sta[:, tau].reshape(ns_x, ns_y),
                                    interpolation='nearest',
                                    cmap=plt.cm.coolwarm,
                                    clim=clim)
                ax[tau].set_xticks([])
                ax[tau].set_yticks([])
                ax[tau].set_title('$\\tau$ = %.0f ms' %- (-delta_future + tau * kt))
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('intensity', labelpad=40, rotation=270)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()

            #fig.tight_layout()
            fig.subplots_adjust(top=.88)

            plt.suptitle('STA for different time lags\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            return fig

    def plt_deltas_norm(self):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            (ns_x, ns_y) = (Stim() & key).fetch1['ns_x', 'ns_y']
            sta = (self & key).fetch1['sta']
            ns, nt = sta.shape
            delta_future, delta_past = (self & key).fetch1['tfuture', 'tpast']
            kt = (delta_past + delta_future) / nt

            sta_z = (sta - np.mean(sta[0:ns_x, :])) / abs(sta).max()

            fig, axarr = plt.subplots(2, int(nt / 2))
            ax = axarr.flatten()
            clim = (sta_z.min(), sta_z.max())

            for tau in range(int(len(ax))):
                im = ax[tau].imshow(sta_z[:, tau].reshape(ns_x, ns_y),
                                    interpolation='nearest',
                                    cmap=plt.cm.coolwarm,
                                    clim=clim)
                ax[tau].set_xticks([])
                ax[tau].set_yticks([])
                ax[tau].set_title('$\\tau$ = %.0f ms' %- (-delta_future + tau * kt))
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('normalized intensity', labelpad=40, rotation=270)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()

            #fig.tight_layout()
            fig.subplots_adjust(top=.88)
            plt.suptitle('Normalized STA for different time lags\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            return fig

    def plt_deltas_sd(self):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            (ns_x, ns_y) = (Stim() & key).fetch1['ns_x', 'ns_y']
            sta = (self & key).fetch1['sta']
            ns, nt = sta.shape
            delta_future, delta_past = (self & key).fetch1['tfuture', 'tpast']
            kt = (delta_past + delta_future) / nt

            sd_map = np.std(sta, 1)
            sta_sd = sta / sd_map[:, None]

            fig, axarr = plt.subplots(2, int(nt / 2))
            ax = axarr.flatten()
            clim = (-np.percentile(sta_sd,90), np.percentile(sta_sd,90))

            for tau in range(int(len(ax))):
                im = ax[tau].imshow(sta_sd[:, tau].reshape(ns_x, ns_y),
                                    interpolation='nearest',
                                    cmap=plt.cm.coolwarm,
                                    clim=clim)
                ax[tau].set_xticks([])
                ax[tau].set_yticks([])
                ax[tau].set_title('$\\tau$ = %.0f ms' %- (-delta_future + tau * kt))
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('s.d. units', labelpad=40, rotation=270)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()

            #fig.tight_layout()
            fig.subplots_adjust(top=.88)
            plt.suptitle('Standard deviation of STA for different time lags\n' + str(exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

    def plt_svd(self):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            (ns_x, ns_y) = (Stim() & key).fetch1['ns_x', 'ns_y']
            rf,kernel = (self & key).fetch1['rf','kernel']
            ws, wt = (self & key).fetch1['ws', 'wt']
            idt,delta_future,delta_past = (self & key).fetch1['idt','tfuture','tpast']
            nt = len(kernel)
            kt = (delta_past + delta_future) / nt
            tau = delta_future - idt*kt


            fig = plt.figure()
            fig.suptitle(' STA at $\Delta$ t: ' + str(tau) + ' ms (upper panel) and SVD (lower panel) \n' + str(
                exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            fig.add_subplot(2, 3, 1)

            im = plt.imshow(rf.reshape(ns_x, ns_y), interpolation='none', cmap=plt.cm.coolwarm, origin='upper')
            cbi = plt.colorbar(im)
            plt.xticks([])
            plt.yticks([])
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbi.locator = tick_locator
            cbi.update_ticks()

            fig.add_subplot(2, 2, 2)
            t = np.linspace(delta_future, -delta_past + delta_future, nt)
            if abs(kernel.min()) > abs(kernel.max()):
                plt.plot(t, scimage.gaussian_filter(kernel, .7), color=curpal[0], linewidth=4)
            else:
                plt.plot(t, scimage.gaussian_filter(kernel, .7), color=curpal[2], linewidth=4)

            plt.locator_params(axis='y', nbins=4)
            ax = fig.gca()
            ax.set_xticklabels([])
            ax.set_xlim([delta_future, -delta_past + delta_future])
            plt.ylabel('stimulus intensity', labelpad=20)
            plt.locator_params(axis='x', nbins=6)
            plt.locator_params(axis='y', nbins=4)

            fig.add_subplot(2, 3, 4)
            im = plt.imshow(ws.reshape(ns_x, ns_y), interpolation='none', cmap=plt.cm.coolwarm, origin='upper')
            cbi = plt.colorbar(im)
            plt.xticks([])
            plt.yticks([])
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbi.locator = tick_locator
            cbi.update_ticks()
            plt.xticks([])
            plt.yticks([])

            fig.add_subplot(2, 2, 4)

            if abs(wt.min()) > abs(wt.max()):
                plt.plot(t, scimage.gaussian_filter(wt, .7), color=curpal[0], linewidth=4)
            else:
                plt.plot(t, scimage.gaussian_filter(wt, .7), color=curpal[2], linewidth=4)

            plt.locator_params(axis='y', nbins=4)
            ax = fig.gca()
            ax.set_xlim([delta_future, -delta_past + delta_future])
            plt.xlabel('time [ms]', labelpad=10)
            plt.ylabel('stimulus intensity', labelpad=20)

            plt.locator_params(axis='x', nbins=6)
            plt.locator_params(axis='y', nbins=4)

            plt.subplots_adjust(top=.8)

            return fig

    def plt_spacetime(self):

        plt.rcParams.update(
            {'figure.figsize': (12, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )

        for key in self.project().fetch.as_dict:
            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            #(ns_x, ns_y) = (Stim() & key).fetch1['ns_x', 'ns_y']
            sta = (self & key).fetch1['sta']
            delta_future,delta_past = (self & key).fetch1['tfuture','tpast']
            ns, nt = sta.shape

            fig, ax = plt.subplots()

            ax.imshow(np.repeat(sta, 10, axis=1).T[::-1], cmap=plt.cm.coolwarm)
            ax.set_xlabel('space')
            ax.set_ylabel('time [ms]', labelpad=20)
            ax.set_yticks(np.linspace(4, nt * 10 - 4, nt))
            ax.set_yticklabels(np.linspace(-delta_past + delta_future, delta_future, nt).astype(int))

            #fig.tight_layout()
            fig.subplots_adjust(top=.88)

            plt.suptitle('Spacetime STA\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            return fig

@schema
class Stc(dj.Computed):
    definition="""
    -> Stim
    -> Sta
    ---
    stc_pca     :longblob       # array (ns x nt) with first eigenvector of stc as column at each time lag
    stc_ev      :longblob       # array (ns x nt) with eigenvalues of stc as column at each time lag
    """

    def _make_tuples(self, key):

        fs = (Recording() & key).fetch1['fs']

        rec_len = (Spikes() & key).fetch1['rec_len']
        spiketimes = (Spikes() & key).fetch1['spiketimes']
        triggertimes = (Trigger() & key).fetch1['triggertimes']

        freq = (StimMeta() & key).fetch1['freq']

        ns_x, ns_y = (Stim() & key).fetch1['ns_x', 'ns_y']

        Sc = (Stim() & key).fetch1['sc']
        sta = (Sta() & key).fetch1['sta']
        delta_future,delta_past = (Sta() & key).fetch1['tfuture','tpast']
        ns,nt = sta.shape
        npast = int(delta_past * fs * 1e-3)
        nfuture = int(delta_future * fs * 1e-3)

        kn = int((npast + nfuture) / (nt))  # sampling of stimulus in 100 ms steps
        ntrigger = int(len(triggertimes))


        npast = int(delta_past * fs * 1e-3)
        nfuture = int(delta_future * fs * 1e-3)

        spiketimes = spiketimes[spiketimes > triggertimes[0] + npast]
        spiketimes = spiketimes[spiketimes < triggertimes[ntrigger - 1] + int(fs / freq) - 1]
        nspikes = int(len(spiketimes))
        Scut = Sc[:, 0:ntrigger]

        stimInd = np.zeros(rec_len).astype(int) - 1
        for n in range(ntrigger - 1):
            stimInd[triggertimes[n]:triggertimes[n + 1] - 1] += int(n + 1)
        stimInd[triggertimes[ntrigger - 1]:triggertimes[ntrigger - 1] + (fs / freq) - 1] += int(ntrigger)

        ste = np.zeros([ns, nt, nspikes])

        for sp in range(nspikes):
            for t in range(-nfuture, npast, kn):
                ste[:, int((t + nfuture) / kn), sp] = np.array(Scut[:, stimInd[spiketimes[sp] - t]])

        stc = np.zeros((ns, ns, nt))
        stc_pca = np.zeros(sta.shape)
        stc_ev = np.zeros(sta.shape)
        for tau in range(nt):
            stc[:, :, tau] = np.dot((ste[:, tau, :] - sta[:, tau, None]),
                                    (ste[:, tau, :] - sta[:, tau, None]).T) / nspikes
            ev, evec = np.linalg.eig(stc[:, :, tau])

            stc_pca[:, tau] = np.mean(evec[:, 0:2], 1)  # keep first pc
            stc_ev[:, tau] = ev

        self.insert1(dict(key, stc_pca = stc_pca, stc_ev = stc_ev))

    def plt_deltas(self):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )

        for key in self.project().fetch.as_dict:


            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            (ns_x, ns_y) = (Stim() & key).fetch1['ns_x', 'ns_y']
            sta = (Sta() & key).fetch1['sta']
            ns, nt = sta.shape
            delta_future, delta_past = (Sta() & key).fetch1['tfuture', 'tpast']
            kt = (delta_past + delta_future) / nt

            stc_pca = (self & key).fetch1['stc_pca']

            fig, axarr = plt.subplots(2, int(nt / 2))
            ax = axarr.flatten()
            clim = (stc_pca.min(), stc_pca.max())

            for tau in range(int(len(ax))):
                im = ax[tau].imshow(stc_pca[:, tau].reshape(ns_x, ns_y),
                                    interpolation='nearest',
                                    cmap=plt.cm.coolwarm,
                                    clim=clim)
                ax[tau].set_xticks([])
                ax[tau].set_yticks([])
                ax[tau].set_title('$\\tau$ = %.0f ms' %-(-delta_future + tau * kt))
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('intensity', labelpad=40, rotation=270)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()

            #fig.tight_layout()
            fig.subplots_adjust(top=.88)
            plt.suptitle('First PC of STC for different time lags\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            return fig

    def plt_spacetime(self):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )

        for key in self.project().fetch.as_dict:
            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            # (ns_x, ns_y) = (Stim() & key).fetch1['ns_x', 'ns_y']
            stc_pca = (self & key).fetch1['stc_pca']
            delta_future, delta_past = (Sta() & key).fetch1['tfuture', 'tpast']
            ns, nt = stc_pca.shape

            fig, ax = plt.subplots()
            fig.tight_layout()

            ax.imshow(np.repeat(stc_pca, 10, axis=1).T[::-1], cmap=plt.cm.coolwarm)
            ax.set_xlabel('space')
            ax.set_ylabel('time [ms]', labelpad=20)
            ax.set_yticks(np.linspace(4, nt * 10 - 4, nt))
            ax.set_yticklabels(np.linspace(-delta_past + delta_future, delta_future, nt).astype(int))

            fig.tight_layout()

            plt.suptitle('First PC of spacetime STC\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            return fig

    def plt_eigenvalues(self):

        plt.rcParams.update(
            {'figure.figsize': (12, 6),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )

        for key in self.project().fetch.as_dict:
            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            delta_future, delta_past = (Sta() & key).fetch1['tfuture', 'tpast']

            stc_ev = (self & key).fetch1['stc_ev']
            ns,nt = stc_ev.shape

            kt = (delta_past + delta_future) / nt

            pals = sns.cubehelix_palette(nt, start=.2, rot=-1)

            fig, ax = plt.subplots()


            for tau in range(nt):
                ax.plot(stc_ev[:,tau],'o',label='$\\tau$: %.0f'%-(-delta_future + tau * kt),color=pals[tau])

            ax.set_xlim([-1,ns+1])
            ax.set_xlabel('# Eigenvalue')
            ax.set_ylabel('Variance')
            ax.locator_params(nbins=5)
            ax.legend()

            fig.tight_layout()
            fig.subplots_adjust(top=.88)
            plt.suptitle('Eigenvalues of STC for different time lags\n' + str(exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)
            return fig




@schema
class StimInst(dj.Computed):
    definition="""
    -> Stim
    -> Sta
    ---
    s_inst      :longblob       # array (ns x T)
    """

    def _make_tuples(self, key):

        freq = (StimMeta() & key).fetch1['freq']
        Sc = (Stim() & key).fetch1['sc']
        delta_past, delta_future = (Sta() & key).fetch1['tpast','tfuture']
        wt =(Sta() & key).fetch1['wt']

        nt = len(wt)
        kt = (delta_past + delta_future) /nt

        step = int((1 / freq * 1e3) / kt)
        weights = wt[::step]  # need to get in same resolution as stim freq
        weights_pad = np.vstack((np.zeros(weights[:, None].shape),
                                 weights[:, None]))  # zero-padding to shift origin of weights vector accordingly

        s_inst = scimage.filters.convolve(Sc.T, weights_pad).T

        self.insert1(dict(key,s_inst = s_inst))
@schema
class StaInst(dj.Computed):
    definition="""
    -> StimInst
    -> Spikes
    -> Trigger
    ---
    sta_inst    :longblob   # array (ns x 1) instantaneous linear spatial filter
    y           :longblob  # spike counts vector binned with 1/freq ms

    """

    @property
    def populated_from(self):
        return Recording() & dict(stim_type='bw_noise')

    def _make_tuples(self,key):

        fs = (Recording() & key).fetch1['fs']
        freq = (StimMeta() & key).fetch1['freq']

        rec_len = (Spikes() & key).fetch1['rec_len']
        spiketimes = (Spikes() & key).fetch1['spiketimes']
        triggertimes,ntrigger = (Trigger() & key).fetch1['triggertimes','ntrigger']


        s_inst = (StimInst() & key).fetch1['s_inst']
        s_inst = s_inst[:,0:ntrigger]

        y = []

        for n in range(ntrigger - 1):

            y.append(int(len(spiketimes[(spiketimes > triggertimes[n]) & (spiketimes < triggertimes[n + 1])])))

        y.append(int(len(spiketimes[(spiketimes > triggertimes[ntrigger - 1]) & (spiketimes < triggertimes[ntrigger - 1] + (fs / freq))])))
        y = np.array(y)



        I = np.dot(s_inst, s_inst.T)
        a = np.dot(s_inst, y)

        w_sta = np.linalg.solve(I, a)

        self.insert1(dict(key,sta_inst = w_sta,y=y))


@schema
class StcInst(dj.Computed):
    definition="""
    ->StimInst
    ->StaInst
    ---
    stc_triu        :longblob
    """

    def _make_tuples(self,key):


        ntrigger = (Trigger() & key).fetch1['ntrigger']
        y,sta_inst = (StaInst() & key).fetch1['y','sta_inst']
        s_inst = (StimInst() & key).fetch1['s_inst']
        ste_inst = []

        for t in range(ntrigger):
            if y[t] != 0:
                for sp in range(y[t]):
                    ste_inst.append(s_inst[:, t])
        ste_inst = np.array(ste_inst)

        stc = np.dot((ste_inst - sta_inst).T, (ste_inst - sta_inst)) / y.sum()

        stc_triu = stc[np.triu_indices_from(stc)]



        self.insert1(dict(key,
                            stc_triu = stc_triu
                            ))
@schema
class StcInstPca(dj.Computed):
    definition="""
    ->StcInst
    ---
    stc_highvar     :longblob
    stc_lowvar      :longblob
    stc_ev          :longblob
    """

    def _make_tuples(self,key):

        ns_x, ns_y = (Stim() & key).fetch1['ns_x','ns_y']
        stc_triu = (StcInst() & key).fetch1['stc_triu']

        ns = ns_x*ns_y

        stc_new = np.zeros((ns, ns))
        inds = np.triu_indices_from(stc_new)
        stc_new[inds] = stc_triu
        stc_new[inds[1], inds[0]] = stc_triu

        ev, evec = np.linalg.eig(stc_new)

        fig = plt.figure(figsize=(8,6))

        plt.plot(ev, 'o')
        plt.xlabel('# Eigenvector')
        plt.ylabel('Variance')

        display(fig)

        highvar = float(input('Set ev threshold for high variance components'))
        stc_highvar = []
        for e in np.where(ev > highvar)[0]:
            stc_highvar.append(evec[:, e].astype(float))
        stc_highvar = np.array(stc_highvar)

        lowvar = float(input('Set ev threshold for low variance components'))
        stc_lowvar = []
        for e in np.where(ev < lowvar)[0]:
            stc_lowvar.append(evec[:, e].astype(float))
        stc_lowvar = np.array(stc_lowvar)

        self.insert1(dict(key,
                          stc_highvar=stc_highvar,
                          stc_lowvar=stc_lowvar,
                          stc_ev=ev
                          ))


    def plt_highvar(self):
        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            ns_x, ns_y = (Stim() & key).fetch1['ns_x', 'ns_y']
            sta_inst = (StaInst() & key).fetch1['sta_inst']

            stc_highvar = (self & key).fetch1['stc_highvar']
            stc_ev = (self & key).fetch1['stc_ev']

            for e in range(stc_highvar.shape[0]):
                fig, ax = plt.subplots(1, 2)
                im0 = ax[0].imshow(sta_inst.reshape(ns_x, ns_y), cmap=plt.cm.coolwarm, interpolation='nearest')
                cbar = plt.colorbar(im0, ax=ax[0], shrink=.8)
                tick_locator = ticker.MaxNLocator(nbins=5)
                cbar.locator = tick_locator
                cbar.update_ticks()
                ax[0].set_xticklabels([])
                ax[0].set_yticklabels([])

                ax[0].set_title('$w_{STA}$')

                im1 = ax[1].imshow(stc_highvar[e, :].reshape(ns_x, ns_y), cmap=plt.cm.coolwarm, interpolation='nearest')
                cbar = plt.colorbar(im1, ax=ax[1], shrink=.8)
                tick_locator = ticker.MaxNLocator(nbins=5)
                cbar.locator = tick_locator
                cbar.update_ticks()

                ax[1].set_title('$w_{STC}^{high var},\; \\sigma$: %.1f'%stc_ev[e])
                ax[1].set_xticklabels([])
                ax[1].set_yticklabels([])
                fig.tight_layout()
                fig.subplots_adjust(top=.88)

                plt.suptitle('High Var Components of instantaneous STC\n' + str(
                    exp_date) + ': ' + eye + ': ' + fname,
                             fontsize=16)
                return fig

    def plt_lowvar(self):
        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            ns_x, ns_y = (Stim() & key).fetch1['ns_x', 'ns_y']
            sta_inst = (StaInst() & key).fetch1['sta_inst']
            ns = ns_x * ns_y

            stc_lowvar = (self & key).fetch1['stc_lowvar']
            stc_ev = (self & key).fetch1['stc_ev']


            for e in range(stc_lowvar.shape[0]):
                fig, ax = plt.subplots(1, 2)
                im0 = ax[0].imshow(sta_inst.reshape(ns_x, ns_y), cmap=plt.cm.coolwarm, interpolation='nearest')
                cbar = plt.colorbar(im0, ax=ax[0], shrink=.8)
                tick_locator = ticker.MaxNLocator(nbins=5)
                cbar.locator = tick_locator
                cbar.update_ticks()
                ax[0].set_xticklabels([])
                ax[0].set_yticklabels([])

                ax[0].set_title('$w_{STA}$')

                im1 = ax[1].imshow(stc_lowvar[e, :].reshape(ns_x, ns_y), cmap=plt.cm.coolwarm,
                                   interpolation='nearest')
                cbar = plt.colorbar(im1, ax=ax[1], shrink=.8)
                tick_locator = ticker.MaxNLocator(nbins=5)
                cbar.locator = tick_locator
                cbar.update_ticks()

                ax[1].set_title('$w_{STC}^{low var},\; \\sigma$: %.1f' % stc_ev[ns-(stc_lowvar.shape[0]-e)])
                ax[1].set_xticklabels([])
                ax[1].set_yticklabels([])
                fig.tight_layout()
                fig.subplots_adjust(top=.88)

                plt.suptitle('Low Var Components of instantaneous STC\n' + str(
                    exp_date) + ': ' + eye + ': ' + fname,
                             fontsize=16)
                return fig


    def plt_ev(self):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:
            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            ev = (self & key).fetch1['stc_ev']

            fig = plt.figure(figsize=(8, 6))

            plt.plot(ev, 'o')
            plt.xlabel('# Eigenvector')
            plt.ylabel('Variance')
            plt.xlim([-1,len(ev)+1])

            fig.tight_layout()
            fig.subplots_adjust(top=.88)

            plt.suptitle('Eigenvalues of instantaneous STC\n' + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)
            return fig


@schema
class NonlinInst(dj.Computed):
    definition = """
    -> StimInst
    -> StaInst
    ---
    s1d_sta :longblob   # binned 1-dimensional stimulus projected onto sta axis
    rse_mean    :double # mean of the projected raw stimulus ensemble
    rse_var     :double # variance of the projected raw stimulus ensemble
    p_rse   :longblob   # density along 1d axis of raw stimulus ensemble
    ste_mean    :double # mean of the projected spike-triggered stimulus ensemble
    ste_var     :double # variance of the projected spike-trigger stimulus ensemble
    p_ste   :longblob   # density along 1d axis of spike-triggered stimulus ensemble
    rate    :longblob   # ratio between histograms along 1d stimulus axis
    """

    @property
    def populated_from(self):
        return Recording() & dict(stim_type='bw_noise')

    def _make_tuples(self, key):

        ntrigger =(Trigger() & key).fetch1['ntrigger']
        s_inst =(StimInst() & key).fetch1['s_inst']
        w_sta,y = (StaInst() & key).fetch1['sta_inst','y']


        rse1d = np.dot(w_sta, s_inst)
        nb = 100
        lim = (rse1d.min() - 1, rse1d.max() + 1)
        p_rse, vals = np.histogram(rse1d, bins=nb, range=(lim))
        s1d = vals[0:nb]

        ste1d = []

        for t in range(ntrigger):
            if y[t] != 0:
                for sp in range(y[t]):
                    ste1d.append(rse1d[t])
        ste1d = np.array(ste1d)
        p_ste, vals_ste = np.histogram(ste1d, bins=nb, range=(lim))

        rate = p_ste / p_rse/ nb


        self.insert1(dict(key,
                          s1d_sta=s1d,
                          rse_mean = np.mean(rse1d),
                          rse_var=np.var(rse1d),
                          ste_mean=np.mean(ste1d),
                          ste_var=np.var(ste1d),
                          p_rse=p_rse,
                          p_ste=p_ste,
                          rate=rate
                          ))

    def plt_1dhistograms(self):

        plt.rcParams.update(
            {'figure.figsize': (12, 6),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:
            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']
            nspikes = (Spikes() & key).fetch1['nspikes']
            ntrigger = (Trigger() & key).fetch1['ntrigger']
            s1d =(self & key).fetch1['s1d_sta']
            p_rse,rse_mean,rse_var = (self & key).fetch1['p_rse','rse_mean','rse_var']
            p_ste, ste_mean, ste_var = (self & key).fetch1['p_ste', 'ste_mean', 'ste_var']

            fig, ax = plt.subplots()
            fig.tight_layout()
            fig.subplots_adjust(top=.88)

            lim = (s1d.min(),s1d.max())
            ax.bar(s1d, p_rse / ntrigger, width=.1, label='$p(s)$')
            ax.bar(s1d, p_ste/ nspikes, width=.1, facecolor=curpal[2], label='$p(s|y)$')
            ax.axvline(x=rse_mean, color=curpal[1])
            ax.axvline(x=ste_mean, color=curpal[3])
            ax.set_xlabel('Projection onto STA axis')
            ax.set_ylabel('Probability', labelpad=20)
            ax.legend(fontsize=20)
            ax.set_xlim(lim)
            plt.locator_params('y', nbins=4)

            plt.suptitle('Histogram of the raw and spike-triggered stimulus ensemble\n' + str(exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

    def plt_rate(self):

        plt.rcParams.update(
            {'figure.figsize': (12, 6),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            s1d,rate = (self & key).fetch1['s1d_sta','rate']

            p_ys = np.nan_to_num(rate)

            fig, ax = plt.subplots()

            ax.plot(s1d[p_ys != 0], p_ys[p_ys != 0], 'o', markersize=12)
            ax.set_xlabel('projection onto STA axis')
            ax.set_ylabel('rate $\\frac{s|y}{s}$', labelpad=20)
            plt.locator_params(nbins=4)

            fig.tight_layout()
            fig.subplots_adjust(top=.88,left=.1)

            plt.suptitle('Ratio between STE and RSE densities\n' + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

@schema
class NonlinInstExp(dj.Computed):
    definition = """
    -> NonlinInst
    ---
    aopt    :double     # parameter fit for instantaneous non-linearity of the form a*np.exp(b*x) + c
    bopt    :double     # parameter fit for instantaneous non-linearity of the form a*np.exp(b*x) + c
    copt    :double     # parameter fit for instantaneous non-linearity of the form a*np.exp(b*x) + c
    res     :double     # absolute residuals

    """

    def _make_tuples(self, key):

        nspikes = (Spikes() & key).fetch1['nspikes']
        s1d,rate = (NonlinInst() & key).fetch1['s1d_sta','rate']
        p_ys = np.nan_to_num(rate)

        try:
            popt, pcov = scoptimize.curve_fit(self.non_lin_exp, s1d[p_ys != 0], p_ys[p_ys != 0])

        except Exception as e1:
            print('Exponential fit failed due to:\n', e1)
            popt=(0,0,0)

        aopt, bopt, copt = popt

        res = abs(self.non_lin_exp(s1d[p_ys != 0], aopt, bopt, copt) - p_ys[p_ys != 0]).sum()/nspikes

        self.insert1(dict(key,
                          aopt=aopt,
                          bopt=bopt,
                          copt=copt,
                          res = res))


    def non_lin_exp(self,x,a,b,c):
        return a * np.exp(b * x) + c

    def plt_rate(self):

        plt.rcParams.update(
            {'figure.figsize': (12, 6),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            s1d,rate = (NonlinInst() & key).fetch1['s1d_sta','rate']
            aopt,bopt,copt,res = (self & key).fetch1['aopt','bopt','copt','res']


            p_ys = np.nan_to_num(rate)
            f = self.non_lin_exp(s1d,aopt,bopt,copt)

            fig, ax = plt.subplots()

            ax.plot(s1d[p_ys != 0], p_ys[p_ys != 0], 'o', markersize=12,label='histogramm ratio')
            ax.plot(s1d, f,label='fit',color=curpal[2],linewidth=2)
            ax.set_xlabel('projection onto STA axis')
            ax.set_ylabel('rate $\\frac{s|y}{s}$', labelpad=20)
            plt.locator_params(nbins=4)
            ax.legend()

            fig.tight_layout()
            fig.subplots_adjust(top=.88)

            plt.suptitle('Instantaneous Non-Linearity Estimate: $\\Sigma_{res}$ %.1e\n'%(res) + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

@schema
class NonlinInstSoftmax(dj.Computed):
    definition = """
    -> NonlinInst
    ---
    aopt    :double     # parameter fit for softmax func as instantaneous non-linearity
    topt    :double     # parameter fit for softmax func as instantaneous non-linearity
    res     :double     # absolute residuals

    """

    def _make_tuples(self, key):

        nspikes = (Spikes() & key).fetch1['nspikes']
        s1d,rate = (NonlinInst() & key).fetch1['s1d_sta','rate']
        p_ys = np.nan_to_num(rate)

        try:
            popt, pcov = scoptimize.curve_fit(self.softmax, s1d[p_ys != 0], p_ys[p_ys != 0])

        except Exception as e1:
            print('Exponential fit failed due to:\n', e1)
            popt=(0,0)

        aopt, topt = popt

        res = abs(self.softmax(s1d[p_ys!=0],aopt,topt) - p_ys[p_ys!=0]).sum()/nspikes

        self.insert1(dict(key,
                          aopt=aopt,
                          topt = topt,
                          res = res))

    def softmax(self,x, a, t):
        ex = np.exp(x - a) / t
        sm = ex / ex.sum()

        return sm

    def plt_rate(self):

        plt.rcParams.update(
            {'figure.figsize': (12, 6),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            s1d,rate = (NonlinInst() & key).fetch1['s1d_sta','rate']
            aopt,topt,res = (self & key).fetch1['aopt','topt','res']


            p_ys = np.nan_to_num(rate)
            f = self.softmax(s1d,aopt,topt)

            fig, ax = plt.subplots()

            ax.plot(s1d[p_ys != 0], p_ys[p_ys != 0], 'o', markersize=12,label='histogramm ratio')
            ax.plot(s1d, f,label='fit',color=curpal[2],linewidth=2)
            ax.set_xlabel('projection onto STA axis')
            ax.set_ylabel('rate $\\frac{s|y}{s}$', labelpad=20)
            plt.locator_params(nbins=4)
            ax.legend()

            fig.tight_layout()
            fig.subplots_adjust(top=.88,left=.1)
            plt.suptitle('Instantaneous Non-Linearity Estimate: $\\Sigma_{res}$ %.1e\n'%(res) + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

@schema
class NonlinInstThreshold(dj.Computed):
    definition = """
    -> NonlinInst
    ---
    aopt    :double     # parameter fit for piecewise threshold func as instantaneous non-linearity
    thropt  :double   # parameter fit for piecewise threshold func as instantaneous non-linearity
    res     :double     # absolute residuals

    """

    def _make_tuples(self, key):

        nspikes = (Spikes() & key).fetch1['nspikes']
        s1d,rate = (NonlinInst() & key).fetch1['s1d_sta','rate']
        p_ys = np.nan_to_num(rate)

        try:
            popt, pcov = scoptimize.curve_fit(self.threshold, s1d[p_ys != 0], p_ys[p_ys != 0])

        except Exception as e1:
            print('Exponential fit failed due to:\n', e1)
            popt=(0,0)

        aopt, thropt = popt

        res = abs(self.threshold(s1d[p_ys!=0],aopt,thropt) - p_ys[p_ys!=0]).sum()/nspikes

        self.insert1(dict(key,
                          aopt=aopt,
                          thropt = thropt,
                          res = res))

    def threshold(self,x, a, thr):

        return np.piecewise(x, [x < thr, x >= thr], [0, lambda x: a * x])

    def plt_rate(self):

        plt.rcParams.update(
            {'figure.figsize': (12, 6),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            s1d,rate = (NonlinInst() & key).fetch1['s1d_sta','rate']
            aopt,thropt,res = (self & key).fetch1['aopt','thropt','res']


            p_ys = np.nan_to_num(rate)
            f = self.threshold(s1d,aopt,thropt)

            fig, ax = plt.subplots()

            ax.plot(s1d[p_ys != 0], p_ys[p_ys != 0], 'o', markersize=12,label='histogramm ratio')
            ax.plot(s1d, f,label='fit',color=curpal[2],linewidth=2)
            ax.set_xlabel('projection onto STA axis')
            ax.set_ylabel('rate $\\frac{s|y}{s}$', labelpad=20)
            plt.locator_params(nbins=4)
            ax.legend()

            fig.tight_layout()
            fig.subplots_adjust(top=.88,left=.1)

            plt.suptitle('Instantaneous Non-Linearity Estimate: $\\Sigma_{res}$ %.1e\n'%(res) + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig


@schema
class StaInstRidge(dj.Computed):
    definition="""
    -> StaInst
    -> StimInst
    ---
    sta_inst_ridge  :longblob   # instantaneous RF MAP estimator in a linear gaussian encoding model with ridge regularization
    theta_ridge     :double     # optimal hyperparameter with maximum log-evidence
    sigma_ridge     :double     # encoding noise variance treated as hyperparameter
    log_e           :double     # log-evidence at maximum with optimal hyperparameter set
    """

    def _make_tuples(self,key):

        s_inst = (StimInst() & key).fetch1['s_inst']
        y,w_sta = (StaInst() & key).fetch1['y','sta_inst']
        ntrigger= (Trigger() & key).fetch1['ntrigger']

        s = s_inst[:,0:ntrigger]

        # Init
        theta0 = 1e-6
        sigma0 = np.square(y - np.dot(w_sta, s)).sum() / ntrigger

        ns = s.shape[0]
        T = s.shape[1]

        c_prior0, c_post0, m_post0 = self.params_ridge(theta0, sigma0, s, y)

        theta_r = []
        sigma_r = []
        log_e_list = []
        it = 0

        ## Fixed point iteration converges faster than grad descent on log evidence
        # First iter

        # Update hyperparams
        print('Iter: ', it)

        theta_r.append((ns - theta0 * np.matrix.trace(c_post0)) / np.square(m_post0).sum())

        h7 = y - np.dot(m_post0, s)
        r2 = np.dot(h7, h7.T)
        sigma_r.append(r2 / (T - (1 - theta0 * np.diag(c_post0)).sum()))

        # Update prior and posterior

        c_prior_it, c_post_it, m_post_it = self.params_ridge(theta_r[it], sigma_r[it], s, y)

        log_e_list.append(self.log_e_ridge(theta_r[it], sigma_r[it], s, y))

        dellog_e = 1000
        eps = 10

        # Fixed-point iteration, iter until convergence

        while dellog_e > eps:
            it += 1
            print('Iter: ', it)

            # Update hyperparams according to fixed point rule

            theta_r.append((ns - theta_r[it - 1] * np.matrix.trace(c_post_it)) / np.square(m_post_it).sum())

            h7 = y - np.dot(m_post_it, s)
            r2 = np.dot(h7, h7.T)
            sigma_r.append(r2 / (T - (1 - theta_r[it - 1] * np.diag(c_post_it)).sum()))

            c_prior_it, c_post_it, m_post_it = self.params_ridge(theta_r[it], sigma_r[it], s, y)

            log_e_list.append(self.log_e_ridge(theta_r[it], sigma_r[it], s, y))

            dellog_e = abs(log_e_list[it]) - abs(log_e_list[it - 1])

        c_prior_ridge, c_post_ridge, m_post_ridge = self.params_ridge(theta_r[it],sigma_r[it], s, y)


        self.insert1(dict(key,
                          theta_ridge = theta_r[it],
                          sigma_ridge = sigma_r[it],
                          log_e = log_e_list[it],
                          sta_inst_ridge = m_post_ridge))



    def log_e_ridge(self, theta, sigma, s, y, sign=1):

        """

        :param theta: scalar ridge reg hyperparameter
        :param sigma: scalar encoding noise var
        :param s: array (ns x T) with inst stimulus
        :param y: array(T x 1) with spike counts per stimulus frame
        :param sign: (-1,+1) if -1 negative log-evidence is returned for solving a minimization problem/gradient descent
        :returns
            :return log_e: scalar with sign* log-evidence of a fully Gaussian model
        """

        ns = s.shape[0]
        T = s.shape[1]

        c_prior, c_post, m_post = self.params_ridge(theta, sigma, s, y)

        h3 = np.linalg.solve(c_post.T, c_prior.T).T
        h4sign, h4 = np.linalg.slogdet(h3)
        log_e = sign * (-T * np.log(abs(2 * np.pi * sigma)) / 2 - h4sign * h4 / 2 + np.dot(m_post.T, np.dot(c_post,
                                                                                                            m_post)) / 2 - np.dot(
            y.T, y) / (2 * sigma))

        return log_e

    def params_ridge(self, theta, sigma, s, y):

        """
        Calculate the diagnoal prior and posterior covariance matrix as well as the MAP estimate in a linear gaussian encoding model with ridge regularization

        :param theta: scalar rdige reg hyperparameter
        :param sigma: scalar encoding noise var
        :param s: array instantaneous stimulus as (ns x T)
        :param y: array spike counts vector as array (T x 1)
        :return: c_prior,c_post, m_post
        """

        ns = s.shape[0]
        T = s.shape[1]

        c_prior = np.eye(ns, ns) / theta

        c_post = np.linalg.inv((np.dot(s, s.T) / sigma + theta * np.eye(ns, ns)))

        m_post = np.dot(c_post, np.dot(s, y)) / sigma

        return c_prior, c_post, m_post

    def plt_sta(self):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:
            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            ns_x, ns_y = (Stim() & key).fetch1['ns_x', 'ns_y']
            sta_inst = (StaInst() & key).fetch1['sta_inst']

            # Normalize
            w_sta = sta_inst / abs(sta_inst).max()

            sta_inst_ridge, theta_ridge = (self & key).fetch1['sta_inst_ridge','theta_ridge']

            # Normliaze
            w_ridge = sta_inst_ridge / abs(sta_inst_ridge).max()

            fig, ax = plt.subplots(1, 2)

            im0 = ax[0].imshow(w_sta.reshape(ns_x, ns_y), cmap=plt.cm.coolwarm, interpolation='nearest',clim=(-1,1))
            cbar = plt.colorbar(im0, ax=ax[0], shrink=.8)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()

            ax[0].set_title('$w_{MLE}$', y=1.02, fontsize=20)
            ax[0].set_xticklabels([])
            ax[0].set_yticklabels([])

            im1 = ax[1].imshow(w_ridge.reshape(ns_x, ns_y), cmap=plt.cm.coolwarm, interpolation='nearest',clim=(-1,1))
            cbar = plt.colorbar(im1, ax=ax[1], shrink=.8)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()

            ax[1].set_title('$w_{MAP}^{ridge},\; \\theta = $%.1f' % theta_ridge, y=1.02,
                            fontsize=20)
            ax[1].set_xticklabels([])
            ax[1].set_yticklabels([])

            fig.tight_layout()
            fig.subplots_adjust(top=.85)

            plt.suptitle('Instantaneous STA with Ridge Regression Prior\n' + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

@schema
class NonlinInstRidge(dj.Computed):
    definition = """
    -> StimInst
    -> StaInstRidge
    ---
    s1d_sta :longblob   # binned 1-dimensional stimulus projected onto sta axis
    rse_mean    :double # mean of the projected raw stimulus ensemble
    rse_var     :double # variance of the projected raw stimulus ensemble
    p_rse   :longblob   # density along 1d axis of raw stimulus ensemble
    ste_mean    :double # mean of the projected spike-triggered stimulus ensemble
    ste_var     :double # variance of the projected spike-trigger stimulus ensemble
    p_ste   :longblob   # density along 1d axis of spike-triggered stimulus ensemble
    rate    :longblob   # ratio between histograms along 1d stimulus axis
    """

    @property
    def populated_from(self):
        return Recording() & dict(stim_type='bw_noise')

    def _make_tuples(self, key):

        ntrigger =(Trigger() & key).fetch1['ntrigger']
        s_inst =(StimInst() & key).fetch1['s_inst']
        y = (StaInst() & key).fetch1['y']

        w_sta = (StaInstRidge() & key).fetch1['sta_inst_ridge']


        rse1d = np.dot(w_sta, s_inst)
        nb = 100
        lim = (rse1d.min() - 1, rse1d.max() + 1)
        p_rse, vals = np.histogram(rse1d, bins=nb, range=(lim))
        s1d = vals[0:nb]

        ste1d = []

        for t in range(ntrigger):
            if y[t] != 0:
                for sp in range(y[t]):
                    ste1d.append(rse1d[t])
        ste1d = np.array(ste1d)
        p_ste, vals_ste = np.histogram(ste1d, bins=nb, range=(lim))

        rate = p_ste / p_rse/ nb


        self.insert1(dict(key,
                          s1d_sta=s1d,
                          rse_mean = np.mean(rse1d),
                          rse_var=np.var(rse1d),
                          ste_mean=np.mean(ste1d),
                          ste_var=np.var(ste1d),
                          p_rse=p_rse,
                          p_ste=p_ste,
                          rate=rate
                          ))

    def plt_1dhistograms(self):

        plt.rcParams.update(
            {'figure.figsize': (12, 6),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:
            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']
            nspikes = (Spikes() & key).fetch1['nspikes']
            ntrigger = (Trigger() & key).fetch1['ntrigger']
            s1d =(self & key).fetch1['s1d_sta']
            p_rse,rse_mean,rse_var = (self & key).fetch1['p_rse','rse_mean','rse_var']
            p_ste, ste_mean, ste_var = (self & key).fetch1['p_ste', 'ste_mean', 'ste_var']

            fig, ax = plt.subplots()
            fig.tight_layout()
            fig.subplots_adjust(top=.88)

            lim = (s1d.min(),s1d.max())
            ax.bar(s1d, p_rse / ntrigger, width=.1, label='$p(s)$')
            ax.bar(s1d, p_ste/ nspikes, width=.1, facecolor=curpal[2], label='$p(s|y)$')
            ax.axvline(x=rse_mean, color=curpal[1])
            ax.axvline(x=ste_mean, color=curpal[3])
            ax.set_xlabel('Projection onto STA axis')
            ax.set_ylabel('Probability', labelpad=20)
            ax.legend(fontsize=20)
            ax.set_xlim(lim)
            plt.locator_params('y', nbins=4)

            plt.suptitle('Histogram of the raw and spike-triggered stimulus ensemble\n' + str(exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

    def plt_rate(self):

        plt.rcParams.update(
            {'figure.figsize': (12, 6),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            s1d,rate = (self & key).fetch1['s1d_sta','rate']

            p_ys = np.nan_to_num(rate)

            fig, ax = plt.subplots()

            ax.plot(s1d[p_ys != 0], p_ys[p_ys != 0], 'o', markersize=12)
            ax.set_xlabel('projection onto STA axis')
            ax.set_ylabel('rate $\\frac{s|y}{s}$', labelpad=20)
            plt.locator_params(nbins=4)

            fig.tight_layout()
            fig.subplots_adjust(top=.88,left=.1)

            plt.suptitle('Ratio between STE and RSE densities\n' + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

@schema
class NonlinInstExpRidge(dj.Computed):
    definition = """
    -> NonlinInstRidge
    ---
    aopt    :double     # parameter fit for instantaneous non-linearity of the form a*np.exp(b*x) + c
    bopt    :double     # parameter fit for instantaneous non-linearity of the form a*np.exp(b*x) + c
    copt    :double     # parameter fit for instantaneous non-linearity of the form a*np.exp(b*x) + c
    res     :double     # absolute residuals

    """

    def _make_tuples(self, key):

        nspikes = (Spikes() & key).fetch1['nspikes']
        s1d,rate = (NonlinInstRidge() & key).fetch1['s1d_sta','rate']
        p_ys = np.nan_to_num(rate)

        try:
            popt, pcov = scoptimize.curve_fit(self.non_lin_exp, s1d[p_ys != 0], p_ys[p_ys != 0])

        except Exception as e1:
            print('Exponential fit failed due to:\n', e1)
            popt=(0,0,0)

        aopt, bopt, copt = popt

        res = abs(self.non_lin_exp(s1d[p_ys != 0], aopt, bopt, copt) - p_ys[p_ys != 0]).sum()/nspikes

        self.insert1(dict(key,
                          aopt=aopt,
                          bopt=bopt,
                          copt=copt,
                          res = res))


    def non_lin_exp(self,x,a,b,c):
        return a * np.exp(b * x) + c

    def plt_rate(self):

        plt.rcParams.update(
            {'figure.figsize': (12, 6),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            s1d,rate = (NonlinInst() & key).fetch1['s1d_sta','rate']
            aopt,bopt,copt,res = (self & key).fetch1['aopt','bopt','copt','res']


            p_ys = np.nan_to_num(rate)
            f = self.non_lin_exp(s1d,aopt,bopt,copt)

            fig, ax = plt.subplots()

            ax.plot(s1d[p_ys != 0], p_ys[p_ys != 0], 'o', markersize=12,label='histogramm ratio')
            ax.plot(s1d, f,label='fit',color=curpal[2],linewidth=2)
            ax.set_xlabel('projection onto STA axis')
            ax.set_ylabel('rate $\\frac{s|y}{s}$', labelpad=20)
            plt.locator_params(nbins=4)
            ax.legend()

            fig.tight_layout()
            fig.subplots_adjust(top=.88)

            plt.suptitle('Instantaneous Non-Linearity Estimate: $\\Sigma_{res}$ %.1e\n'%(res) + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

@schema
class NonlinInstSoftmaxRidge(dj.Computed):
    definition = """
    -> NonlinInstRidge
    ---
    aopt    :double     # parameter fit for softmax func as instantaneous non-linearity
    topt    :double     # parameter fit for softmax func as instantaneous non-linearity
    res     :double     # absolute residuals

    """

    def _make_tuples(self, key):

        nspikes = (Spikes() & key).fetch1['nspikes']
        s1d,rate = (NonlinInstRidge() & key).fetch1['s1d_sta','rate']
        p_ys = np.nan_to_num(rate)

        try:
            popt, pcov = scoptimize.curve_fit(self.softmax, s1d[p_ys != 0], p_ys[p_ys != 0])

        except Exception as e1:
            print('Exponential fit failed due to:\n', e1)
            popt=(0,0)

        aopt, topt = popt

        res = abs(self.softmax(s1d[p_ys!=0],aopt,topt) - p_ys[p_ys!=0]).sum()/nspikes

        self.insert1(dict(key,
                          aopt=aopt,
                          topt = topt,
                          res = res))

    def softmax(self,x, a, t):
        ex = np.exp(x - a) / t
        sm = ex / ex.sum()

        return sm

    def plt_rate(self):

        plt.rcParams.update(
            {'figure.figsize': (12, 6),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            s1d,rate = (NonlinInst() & key).fetch1['s1d_sta','rate']
            aopt,topt,res = (self & key).fetch1['aopt','topt','res']


            p_ys = np.nan_to_num(rate)
            f = self.softmax(s1d,aopt,topt)

            fig, ax = plt.subplots()

            ax.plot(s1d[p_ys != 0], p_ys[p_ys != 0], 'o', markersize=12,label='histogramm ratio')
            ax.plot(s1d, f,label='fit',color=curpal[2],linewidth=2)
            ax.set_xlabel('projection onto STA axis')
            ax.set_ylabel('rate $\\frac{s|y}{s}$', labelpad=20)
            plt.locator_params(nbins=4)
            ax.legend()

            fig.tight_layout()
            fig.subplots_adjust(top=.88,left=.1)
            plt.suptitle('Instantaneous Non-Linearity Estimate: $\\Sigma_{res}$ %.1e\n'%(res) + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

@schema
class NonlinInstThresholdRidge(dj.Computed):
    definition = """
    -> NonlinInstRidge
    ---
    aopt    :double     # parameter fit for piecewise threshold func as instantaneous non-linearity
    thropt  :double   # parameter fit for piecewise threshold func as instantaneous non-linearity
    res     :double     # absolute residuals

    """

    def _make_tuples(self, key):

        nspikes = (Spikes() & key).fetch1['nspikes']
        s1d,rate = (NonlinInstRidge() & key).fetch1['s1d_sta','rate']
        p_ys = np.nan_to_num(rate)

        try:
            popt, pcov = scoptimize.curve_fit(self.threshold, s1d[p_ys != 0], p_ys[p_ys != 0])

        except Exception as e1:
            print('Exponential fit failed due to:\n', e1)
            popt=(0,0)

        aopt, thropt = popt

        res = abs(self.threshold(s1d[p_ys!=0],aopt,thropt) - p_ys[p_ys!=0]).sum()/nspikes

        self.insert1(dict(key,
                          aopt=aopt,
                          thropt = thropt,
                          res = res))

    def threshold(self,x, a, thr):

        return np.piecewise(x, [x < thr, x >= thr], [0, lambda x: a * x])

    def plt_rate(self):

        plt.rcParams.update(
            {'figure.figsize': (12, 6),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            s1d,rate = (NonlinInst() & key).fetch1['s1d_sta','rate']
            aopt,thropt,res = (self & key).fetch1['aopt','thropt','res']


            p_ys = np.nan_to_num(rate)
            f = self.threshold(s1d,aopt,thropt)

            fig, ax = plt.subplots()

            ax.plot(s1d[p_ys != 0], p_ys[p_ys != 0], 'o', markersize=12,label='histogramm ratio')
            ax.plot(s1d, f,label='fit',color=curpal[2],linewidth=2)
            ax.set_xlabel('projection onto STA axis')
            ax.set_ylabel('rate $\\frac{s|y}{s}$', labelpad=20)
            plt.locator_params(nbins=4)
            ax.legend()

            fig.tight_layout()
            fig.subplots_adjust(top=.88,left=.1)

            plt.suptitle('Instantaneous Non-Linearity Estimate: $\\Sigma_{res}$ %.1e\n'%(res) + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

@schema
class StaInstArd(dj.Computed):
    definition="""
    ->StaInst
    ->StimInst
    ---
    sta_inst_ard  :longblob   # instantaneous RF MAP estimator in a linear gaussian encoding model with ard prior covariance matrix
    theta_ard     :longblob   # optimal hyperparameter with maximum log-evidence
    sigma_ard     :double     # encoding noise variance treated as hyperparameter
    log_e         :double     # log-evidence at maximum with optimal hyperparameter set
    """

    def _make_tuples(self, key):

        s_inst = (StimInst() & key).fetch1['s_inst']
        y, w_sta = (StaInst() & key).fetch1['y', 'sta_inst']
        ntrigger = (Trigger() & key).fetch1['ntrigger']
        theta_ridge,sigma_ridge = (StaInstRidge() & key).fetch1['theta_ridge','sigma_ridge']

        s = s_inst[:, 0:ntrigger]

        # Init


        ns = s.shape[0]
        T = s.shape[1]

        theta0 = np.repeat(theta_ridge, ns)
        sigma0 = sigma_ridge

        c_prior0, c_post0, m_post0 = self.params_ard(theta0, sigma0, s, y)

        theta_ard = []
        sigma_ard = []
        log_e_list = []
        it = 0

        ## Fixed point iteration converges faster than grad descent on log evidence
        # First iter

        # Update hyperparams
        print('Iter: ', it)

        theta_ard.append((1 - theta0 * np.diag(c_post0)) / np.square(m_post0))

        h7 = y - np.dot(m_post0, s)
        r2 = np.dot(h7, h7.T)
        sigma_ard.append(r2 / (T - (1 - theta0 * np.diag(c_post0)).sum()))

        # Update prior and posterior

        c_prior_it, c_post_it, m_post_it = self.params_ard(theta_ard[it], sigma_ard[it], s, y)

        log_e_list.append(self.log_e_ard(theta_ard[it], sigma_ard[it], s, y))

        dellog_e = 1000
        eps=10

        # Fixed-point iteration, iter until convergence

        while dellog_e > eps:
            it += 1
            print('Iter: ', it)

            # Update hyperparams according to fixed point rule

            theta_ard.append((1 - theta_ard[it - 1] * np.diag(c_post_it)) / np.square(m_post_it))

            h7 = y - np.dot(m_post_it, s)
            r2 = np.dot(h7, h7.T)
            sigma_ard.append(r2 / (T - (1 - theta_ard[it - 1] * np.diag(c_post_it)).sum()))

            c_prior_it, c_post_it, m_post_it = self.params_ard(theta_ard[it], sigma_ard[it], s, y)

            log_e_list.append(self.log_e_ard(theta_ard[it], sigma_ard[it], s, y))

            dellog_e = abs(log_e_list[it]) - abs(log_e_list[it - 1])

            c_prior_ard, c_post_ard, m_post_ard = self.params_ard(theta_ard[it], sigma_ard[it], s, y)

        self.insert1(dict(key,
                          theta_ard=theta_ard[it],
                          sigma_ard=sigma_ard[it],
                          log_e=log_e_list[it],
                          sta_inst_ard=m_post_ard))



    def params_ard(self,theta, sigma, s, y):


        ns = s.shape[0]
        T = s.shape[1]

        c_prior = np.diag(1 / theta)

        c_post = np.linalg.inv((np.dot(s, s.T) / sigma + np.diag(theta) * np.eye(ns, ns)))

        m_post = np.dot(c_post, np.dot(s, y)) / sigma

        return c_prior, c_post, m_post

    def log_e_ard(self,theta, sigma, s, y, sign=1):

        ns = s.shape[0]
        T = s.shape[1]

        c_prior, c_post, m_post = self.params_ard(theta, sigma, s, y)

        h3 = np.linalg.solve(c_post.T, c_prior.T).T
        h4sign, h4 = np.linalg.slogdet(h3)
        log_e = sign * (-T * np.log(abs(2 * np.pi * sigma)) / 2 - h4sign * h4 / 2 + np.dot(m_post.T, np.dot(c_post,
                                                                                                            m_post)) / 2 - np.dot(
            y.T, y) / (2 * sigma))

        return log_e

    def plt_sta(self):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            ns_x, ns_y = (Stim() & key).fetch1['ns_x', 'ns_y']
            sta_inst = (StaInst() & key).fetch1['sta_inst']

            # Normalize
            w_sta = sta_inst / abs(sta_inst).max()

            sta_inst_ard, sigma_ard = (self & key).fetch1['sta_inst_ard', 'sigma_ard']

            # Normliaze
            w_ard = sta_inst_ard / abs(sta_inst_ard).max()

            fig, ax = plt.subplots(1, 2)

            im0 = ax[0].imshow(w_sta.reshape(ns_x, ns_y), cmap=plt.cm.coolwarm, interpolation='nearest', clim=(-1, 1))
            cbar = plt.colorbar(im0, ax=ax[0], shrink=.8)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()

            ax[0].set_title('$w_{MLE}$', y=1.02, fontsize=20)
            ax[0].set_xticklabels([])
            ax[0].set_yticklabels([])

            im1 = ax[1].imshow(w_ard.reshape(ns_x, ns_y), cmap=plt.cm.coolwarm, interpolation='nearest', clim=(-1, 1))
            cbar = plt.colorbar(im1, ax=ax[1], shrink=.8)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()

            ax[1].set_title('$w_{MAP}^{ARD},\; \\sigma^2 = $%.1f' % (sigma_ard), y=1.02, fontsize=20)
            ax[1].set_xticklabels([])
            ax[1].set_yticklabels([])

            fig.tight_layout()
            fig.subplots_adjust(top=.85)

            plt.suptitle('Instantaneous STA with ARD Prior\n' + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

    def plt_cprior(self):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            theta = (self & key).fetch1['theta_ard']

            c_prior = np.diag(1/theta)

            fig, ax = plt.subplots()

            im = ax.imshow(c_prior,cmap = plt.cm.Greys_r)
            cbar = plt.colorbar(im, shrink=.9)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()

            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            fig.tight_layout()
            fig.subplots_adjust(top=.88)

            plt.suptitle('Prior ARD covariance matrix\n' + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig

@schema
class PredStaInst(dj.Computed):
    definition="""
    -> StaInst
    -> StimInst
    ---
    r       :longblob                           # predicted rate
    k       :double                             # split for cross-validation
    rho     :double                             # mean correlation coefficient
    res     :double                             # mean ordinaray test error
    nl_type :enum('exp','sm','thr','none')      # type of rectifying non-linearity used, selected as best fitting from those three
    """

    def _make_tuples(self,key):

        ntrigger = (Trigger() & key).fetch1['ntrigger']

        # fetch spike counts
        y = (StaInst() & key).fetch1['y']

        # fetch stimulus

        s_inst = (StimInst() & key).fetch1['s_inst']
        s = s_inst[:, 0:ntrigger]

        # fetch best fitting non-linearity:
        res_exp = (NonlinInstExp() & key).fetch1['res']
        res_sm = (NonlinInstSoftmax() & key).fetch1['res']
        res_thr = (NonlinInstThreshold() & key).fetch1['res']

        res = np.array([res_exp, res_sm, res_thr])

        id_nl = res.argmin()

        if id_nl == 0:
            aopt, bopt, copt = (NonlinInstExp() & key).fetch1['aopt', 'bopt', 'copt']

            def non_lin(x):
                return aopt * np.exp(bopt * x) + copt

            nl_type = 'exp'

        elif id_nl == 1:
            aopt, topt = (NonlinInstSoftmax() & key).fetch1['aopt', 'topt']

            def non_lin(x):
                ex = np.exp(x - aopt) /topt
                sm = ex / ex.sum()

            nl_type = 'sm'

        elif id_nl == 2:
            aopt, topt = (NonlinInstThreshold() & key).fetch1['aopt', 'thropt']

            def non_lin(x):
                return np.piecewise(x, [x < topt, x >= topt], [0, lambda x: aopt * x])

            nl_type = 'thr'
        else:
            print('Optimal non-linearity not found! Using linear prediction')

            def non_lin(x):
                return x

        k_fold = 10

        kf = KFold(ntrigger, n_folds=k_fold, shuffle=False)

        LNG_dict = {}
        LNG_dict.clear()
        LNG_dict['w'] = []
        LNG_dict['r'] = []
        LNG_dict['y_test'] = []
        LNG_dict['pearson_r'] = []
        LNG_dict['err'] = []

        for train, test in kf:
            I = np.dot(s[:, train], s[:, train].T)
            a = np.dot(s[:, train], y[train])

            w = np.linalg.solve(I, a)

            LNG_dict['w'].append(w)

            LNG_dict['y_test'].append(y[test])

            r0 = np.dot(w, s[:, test])
            r = non_lin(r0)

            LNG_dict['r'].append(r)

            err = np.square(y[test] / ntrigger - r).sum() / len((test))

            LNG_dict['err'].append(err)

            LNG_dict['pearson_r'].append(np.corrcoef(r, y[test])[0, 1])

        LNG_df = pd.DataFrame(LNG_dict)

        r_all = np.array([])

        for ix, row in LNG_df.iterrows():
            r_all = np.hstack((r_all, row.r))

        print(key)
        display(LNG_df)

        self.insert1(dict(key,
                            r = r_all,
                            k = k_fold,
                            rho = np.nanmean(LNG_df.pearson_r,0),
                            res = np.nanmean(LNG_df.err,0),
                            nl_type = nl_type
        ))

    def plt_pred(self,):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2
             }
        )
        curpal = sns.color_palette()

        for key in self.project().fetch.as_dict:

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            freq = (StimMeta() & key).fetch1['freq']
            ns_x, ns_y = (Stim() & key).fetch1['ns_x', 'ns_y']

            w,y = (StaInst() & key).fetch1['sta_inst','y']
            r_all = (PredStaInst() & key).fetch1['r']
            rho,k_fold = (PredStaInst() & key).fetch1['rho','k']

            start = 200
            end = 400
            t = np.linspace(start / freq, end / freq, end - start)

            fig = plt.figure()
            gs1 = gridsp.GridSpec(2, 1)
            gs1.update(left=.05, right=.5)
            ax0 = plt.subplot(gs1[:, :])
            im = ax0.imshow(w.reshape(ns_x, ns_y), cmap=plt.cm.coolwarm_r, interpolation='nearest')
            cbar = plt.colorbar(im, ax=ax0, shrink=.88)
            ax0.set_xticklabels([])
            ax0.set_yticklabels([])
            ax0.set_title('Instantaneous STA')
            # cbar.set_label('stim intensity', labelpad=20, rotation=270)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()

            gs2 = gridsp.GridSpec(2, 1)
            gs2.update(left=.55, right=.95)
            ax1 = plt.subplot(gs2[0, :])
            # ax1.plot(t,y[start:end],label='prediction')
            ax1.plot(t, y[start:end], label='data')
            ax1.set_xlim([start / freq, end / freq])
            ax1.set_ylabel('spike counts')
            ax1.legend()
            ax1.set_title('$\\rho$ = %.2f' %rho)
            ax1.locator_params(nbins=4)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.set_yticklabels([])

            ax2 = plt.subplot(gs2[1, :], sharex=ax1)
            ax2.plot(t, r_all[start:end], label='rate $\lambda$')
            ax2.legend()
            ax2.set_xlabel('time [s]')
            ax2.set_ylabel('firing rate')
            ax2.set_xlim([start / freq, end / freq])
            ax2.set_yticklabels([])

            ax2.locator_params(nbins=4)

            plt.suptitle('Instantaneous STA with k-fold = %.0f cross-validation\n'%k_fold + str(
                exp_date) + ': ' + eye + ': ' + fname,
                         fontsize=16)

            return fig
















def addEntry(animal_id,sex,date_of_birth,exp_date,experimenter,eye,cell_id,data_folder,rec_type, ch_voltage, ch_trigger,filename):
    """

    :param animal_id: str 'ZK0-yyyy-mm-dd'
    :param sex: str 'F' or 'M'
    :param date_of_birth: str 'yyyy-mm-dd'
    :param exp_date: str 'yyyy-mm-dd'
    :param experimenter str 'tstadler' who did the experiment?
    :param eye: str 'R' or 'L'
    :param cell_id: int 1-16
    :param morph: boolean
    :param type: str 'putative cell type'
    :param data_folder: str '/notebooks/Data_write/Data/Stadler/'
    :param filename: str 'BWNoise'
    :return: adds the given recording to the mysql schema 'ageuler_rgcEphys'
    """

    A = Animal()
    E = Experiment()
    C = Cell()
    R = Recording()

    if (len(A & dict(animal_id=animal_id)) == 0):
        print('Animal new')
        try:
            A.insert1({'animal_id': animal_id, 'sex': sex, 'date_of_birth': date_of_birth})
        except Exception as e1:
            print(e1)
    else:
        print('Animal id already in db.')

    if len(E & dict(animal_id=animal_id, exp_date=exp_date, eye=eye)) == 0:
        print('Experiment new')
        try:
            exp_path = data_folder + exp_date + '/' + eye + '/'
            E.insert1({'animal_id': animal_id, 'exp_date': exp_date, 'eye': eye, 'path': exp_path,'experimenter':experimenter})

        except Exception as e2:
            print(e2)
    else:
        print('Experimental day already in db')

    if len(C & dict(animal_id=animal_id, exp_date=exp_date, cell_id=cell_id)) == 0:
        print('Cell id new')
        try:
            subexp_path = str(cell_id) + '/'
            print('Cell id: ', cell_id)
            morph = bool(int(input('Morphology of this cell was recorded? ')))
            cell_type = str(input('Any guess for the cell type? '))
            C.insert1(
                {'animal_id': animal_id, 'exp_date': exp_date, 'eye': eye, 'cell_id': cell_id, 'folder': subexp_path,
                 'morphology': morph, 'type': cell_type})
        except Exception as e3:
            print(e3)
    else:
        print(('Cell already in db'))

    if len(R & dict(animal_id=animal_id, exp_date=exp_date, eye=eye, cell_id=cell_id, filename=filename)) == 0:
        print('Recording new')

        try:
            if 'Noise' in filename:
                R.insert1(
                    {'animal_id': animal_id, 'exp_date': exp_date, 'eye': eye, 'cell_id': cell_id, 'filename': filename,
                     'stim_type': 'bw_noise', 'rec_type': rec_type, 'ch_voltage': ch_voltage, 'ch_trigger': ch_trigger})
            if 'Chirp' in filename:
                R.insert1(
                    {'animal_id': animal_id, 'exp_date': exp_date, 'eye': eye, 'cell_id': cell_id, 'filename': filename,
                     'stim_type': 'chirp', 'rec_type': rec_type, 'ch_voltage': ch_voltage, 'ch_trigger': ch_trigger})
            if 'DS' in filename:
                R.insert1(
                    {'animal_id': animal_id, 'exp_date': exp_date, 'eye': eye, 'cell_id': cell_id, 'filename': filename,
                     'stim_type': 'ds', 'rec_type': rec_type, 'ch_voltage': ch_voltage, 'ch_trigger': ch_trigger})
            if 'ON' in filename:
                R.insert1(
                    {'animal_id': animal_id, 'exp_date': exp_date, 'eye': eye, 'cell_id': cell_id, 'filename': filename,
                     'stim_type': 'on_off', 'rec_type': rec_type, 'ch_voltage': ch_voltage, 'ch_trigger': ch_trigger})
            else:
                R.insert1(
                    {'animal_id': animal_id, 'exp_date': exp_date, 'eye': eye, 'cell_id': cell_id, 'filename': filename,
                     'stim_type': 'unknown', 'rec_type': rec_type, 'ch_voltage': ch_voltage, 'ch_trigger': ch_trigger})
        except Exception as e4:
            print(e4)
    else:
        print('Recording already in db')















