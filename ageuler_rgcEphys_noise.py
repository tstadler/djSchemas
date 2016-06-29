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
        self.insert1(dict(key, triggertimes=triggertimes))

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

    ->Spikes
    ->Trigger
    ->StimMeta
    ---
    s   	: longblob		# stimulus as a (ns x t) array, where ns = ns_x x ns_y
    sc   	: longblob		# centered stimulus as a (ns x t) array, where ns = ns_x x ns_y
    swhite 	: longblob		# whitened stimulus as a (ns x t) array, where ns = ns_x x ns_y
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

        self.insert1(dict(key, s = Frames,sc = Fc, swhite=S, t = stim_dim[1], ns_x = int(stim_dim[0]), ns_y = int(stim_dim[1])))

@schema
class Sta(dj.Computed):
    definition = """
        # Calculate the spike-triggered ensemble from noise recording
        -> Recording
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

        sc = (Stim() & key).fetch1['sc']

        ns = int(ns_x*ns_y)
        ntrigger = int(len(triggertimes))

        deltat = 1000  # time lag before spike in [ms]
        delta = int(deltat * fs * 1e-3)
        spiketimes = spiketimes[spiketimes > triggertimes[0] + delta]
        spiketimes = spiketimes[spiketimes < triggertimes[ntrigger - 1] + int(fs / freq) - 1]
        nspikes = int(len(spiketimes))

        Scut = sc[:, 0:ntrigger]

        nt = 11 # number of time steps into the past, delta is deltat/nt

        stimInd = np.zeros(rec_len).astype(int) - 1
        for n in range(ntrigger - 1):
            stimInd[triggertimes[n]:triggertimes[n + 1] - 1] += int(n + 1)

        stimInd[triggertimes[ntrigger - 1]:triggertimes[ntrigger - 1] + (fs / freq) - 1] += int(ntrigger)

        ste = np.zeros([ns, nt - 1, nspikes])
        k = int(delta / (nt - 1))  # sampling of stimulus in 100 ms steps
        future = int(100 * fs * 1e-3)  # check also 100 ms before stimulus
        for sp in range(nspikes):
            for t in range(-future, delta - future, k):
                # print(int(t/k))
                ste[:, int((t + future) / k), sp] = np.array(Scut[:, stimInd[spiketimes[sp] - t]])
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
                            tfuture = 100,
                            tpast = deltat,
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
            deltat = (self & key).fetch1['tpast']
            future = (self & key).fetch1['tfuture']
            ns,nt = sta.shape


            sta = (self & key).fetch1['sta']

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
                ax[tau].set_title('$\\tau$ = %.0f ms' % (future - tau * int(deltat / (nt - 1))))
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('intensity', labelpad=40, rotation=270)

            plt.suptitle('STA for different time lags\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

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














