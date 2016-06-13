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

schema = dj.schema('ageuler_rgcEphys_test3',locals())

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
class Morph(dj.Computed):
    definition="""
    # Reconstructed morphology of the cell as a line-stack
    -> Cell
    ---
    stack       :longblob       # array (scan_z x scan_x x scan_y)
    scan_z      :int            # number of consecutive frames in depth
    scan_y      :int            # scan field y dim
    scan_x      :int            # scan field x dim
    dx          :double         # pixel side length in um
    dy          :double         # pixel side length in um
    zoom        :double         # zoom factor
    scan_size   :double         # side length of scan in um
    df_size_x   :double         # df size in um
    df_size_y   :double         # df size in um
    """

    @property
    def populated_from(self):
        return Cell() & dict(morphology = True)

    def _make_tuples(self,key):

        path = (Experiment() & key).fetch1['path']
        exp_date = (Experiment() & key).fetch1['exp_date']
        folder = (Cell() & key).fetch1['folder']
        cell_id = (Cell() & key).fetch1['cell_id']

        full_path = path + folder + 'linestack.tif'

        stack = tf.imread(full_path)

        # binarize and invert x-axis for proper orientation

        stack_bin = np.zeros(stack.shape)
        for z in range(stack.shape[0]):
            stack_bin[z, :, :] = binarize(stack[z, ::-1, :], threshold=0, copy=True)

        config = ConfigParser()
        config.read(path + folder + 'C' + str(cell_id) + '_' + str(exp_date) + '.ini')
        zoom = config.getfloat('morph', 'zoom')
        scan_size = 1/ zoom * 71.5  # side length of stack image in um

        scan_x = stack.shape[1]
        scan_y = stack.shape[2]

        dx_morph = scan_size / scan_x  # morph pixel side length in um
        dy_morph = scan_size / scan_y  # morph pixel side length in um

        morph = np.mean(stack, 0)

        mask = np.ma.masked_where(morph == 0, morph)

        edges0 = np.ma.notmasked_edges(mask, axis=0)
        edges1 = np.ma.notmasked_edges(mask, axis=1)

        dely = edges0[1][0] - edges0[0][0]
        delx = edges1[1][1] - edges1[0][1]

        df_size_x = (delx.max() + 1) * dx_morph
        df_size_y = (dely.max() + 1) * dy_morph

        self.insert1(dict(key, stack = stack_bin, scan_z = stack.shape[0], scan_y = scan_y, scan_x = scan_x,dx=dx_morph, dy=dy_morph, zoom = zoom, scan_size=scan_size,df_size_x = df_size_x,df_size_y = df_size_y))

    def plt_morph(self):

        plt.rcParams.update(
            {'figure.figsize': (15, 10),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': 0,
             'figure.subplot.wspace': .2
             }
        )

        for key in self.project().fetch.as_dict:

            stack = (self & key).fetch1['stack']
            df_size_x, df_size_y = (self & key).fetch1['df_size_x', 'df_size_y']

            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']
            cell_id = (Cell() & key).fetch1['cell_id']

            morph_vert1 = np.mean(stack[::-1], 1)
            morph_vert2 = np.mean(stack[::-1], 2)
            morph = np.mean(stack, 0)
            clim = (0,.01)

            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(morph, clim=clim)

            fig.add_subplot(2, 2, 2)
            plt.imshow(morph_vert1, clim=clim)

            fig.add_subplot(2, 2, 4)
            plt.imshow(morph_vert2, clim=clim)

            ax = fig.get_axes()
            ax[0].annotate('df size in x [um]: %.2f\ndf size in y [um]: %.2f'%(df_size_x,df_size_y),xy = (20,20),fontsize=14)

            plt.suptitle('Linestack\n' + str(exp_date) + ': ' + eye + ': ' + str(cell_id), fontsize=16)

            plt.tight_layout()
            plt.subplots_adjust(top=.8)

            return fig

@schema
class Cut(dj.Computed):

    definition="""
    # Cut soma from Morphology to get DF

    ->Morph
    ---
    stack_wos   :longblob
    dens1       :longblob
    dens2       :longblob
    idx_thr1    :longblob
    idx_thr2    :longblob
    idx_cut     :int
    """

    def _make_tuples(self,key):

        stack = (Morph() & key).fetch1['stack'][::-1]

        morph_vert1 = np.mean(stack, 1)
        morph_vert2 = np.mean(stack, 2)

        ma_vert1 = np.ma.masked_where(morph_vert1 == 0, morph_vert1)
        ma_vert2 = np.ma.masked_where(morph_vert2 == 0, morph_vert2)

        counts1 = ma_vert1.count(axis=1)
        dens1 = counts1 / counts1.sum()

        counts2 = ma_vert2.count(axis=1)
        dens2 = counts2 / counts2.sum()

        idx_thr1 = np.where(dens1 == dens1[dens1 != 0].min())[0]
        idx_thr2 = np.where(dens2 == dens2[dens2 != 0].min())[0]

        fig_c = self.show_cut(stack,idx_thr1,idx_thr2)
        display(fig_c)
        plt.close(fig_c)

        adjust = bool(int(input('Adjust cut off? [Yes:1 , No:0]: ')))

        if adjust:
            fig_d = self.show_density(dens1,dens2,idx_thr1,idx_thr2)
            display(fig_d)
            plt.close(fig_d)

            idx_cut = int(input('Select frame [int] above which everything will be cut off: '))

        else:
            idx_cut1 = idx_thr1.max()
            idx_cut2 = idx_thr2.max()

            idx_cut = np.max([idx_cut1, idx_cut2])

        morph = np.mean(stack[0:idx_cut, :, :], 0)



        self.insert1(dict(key,stack_wos = stack[0:idx_cut,:,:], dens1 = dens1, dens2 = dens2,idx_thr1 = idx_thr1, idx_thr2 = idx_thr2, idx_cut = idx_cut))

    def show_cut(self,stack,idx_thr1,idx_thr2):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2,
             'lines.linewidth': 1
             }
        )

        morph_vert1 = np.mean(stack, 1)
        morph_vert2 = np.mean(stack, 2)

        idx_cut1 = idx_thr1.max()
        idx_cut2 = idx_thr2.max()

        with sns.axes_style({'grid.color': 'r'}):
            fig_cut, ax = plt.subplots(2, 1)
            clim = (0, .01)

            ax[0].imshow(morph_vert1, clim=clim)
            ax[0].set_yticks([idx_cut1])
            ax[0].set_xticks([])

            ax[1].imshow(morph_vert2, clim=clim)
            ax[1].set_yticks([idx_cut2])
            ax[1].set_xticks([])

            fig_cut.tight_layout()
            fig_cut.subplots_adjust(top=.88)

            return fig_cut

    def show_density(self,dens1,dens2,idx_thr1,idx_thr2):

        plt.rcParams.update(
            {'figure.figsize': (15, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2,
             'lines.linewidth': 1
             }
        )

        cur_pal = sns.current_palette()

        cols1 = [cur_pal[0]] * len(dens1)
        for i in idx_thr1:
            cols1[i] = cur_pal[1]

        cols2 = [cur_pal[0]] * len(dens2)
        for i in idx_thr2:
            cols2[i] = cur_pal[1]

        width = .8
        x = np.linspace(0, dens1.shape[0] - width, dens1.shape[0])
        fig, ax = plt.subplots(1, 2, sharey=True)
        ax[0].bar(x, dens1, color=cols1)
        ax[0].set_xlabel('stack height')
        ax[0].set_ylabel('density of non-zero data points', labelpad=20)
        ax[0].set_xlim([0,dens1.shape[0]])

        plt.locator_params(axis='y', nbins=4)

        ax[1].bar(x, dens2, color=cols2)
        ax[1].set_xlabel('stack height')
        ax[1].set_xlim([0, dens2.shape[0]])

        plt.locator_params(axis='y', nbins=4)
        fig.tight_layout()
        fig.subplots_adjust(top=.88)

        return fig

    def plt_cut(self):

        for key in self.project().fetch.as_dict:
            plt.rcParams.update(
                {'figure.figsize': (15, 8),
                 'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'figure.subplot.hspace': .2,
                 'figure.subplot.wspace': .2,
                 'lines.linewidth': 1
                 }
            )
            stack = (Morph() & key).fetch1['stack'][::-1]
            idx_cut = (self & key).fetch1['idx_cut']

            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']
            cell_id = (Cell() & key).fetch1['cell_id']

            morph_vert1 = np.mean(stack, 1)
            morph_vert2 = np.mean(stack, 2)

            with sns.axes_style({'grid.color': 'r'}):
                fig_cut, ax = plt.subplots(2, 1)
                clim = (0, .01)

                ax[0].imshow(morph_vert1, clim=clim)
                ax[0].set_yticks([idx_cut])
                ax[0].set_xticks([])

                ax[1].imshow(morph_vert2, clim=clim)
                ax[1].set_yticks([idx_cut])
                ax[1].set_xticks([])

                fig_cut.tight_layout()
                fig_cut.subplots_adjust(top=.88)

                return fig_cut

    def plt_density(self):

        for key in self.project().fetch.as_dict:

            plt.rcParams.update(
                {'figure.figsize': (15, 8),
                 'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'figure.subplot.hspace': .2,
                 'figure.subplot.wspace': .2,
                 'lines.linewidth': 1
                 }
            )
            dens1,dens2 = (self & key).fetch1['dens1','dens2']
            idx_thr1, idx_thr2 = (self & key).fetch1['idx_thr1','idx_thr2']

            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']
            cell_id = (Cell() & key).fetch1['cell_id']

            cur_pal = sns.current_palette()

            cols1 = [cur_pal[0]] * len(dens1)
            for i in idx_thr1:
                cols1[i] = cur_pal[1]

            cols2 = [cur_pal[0]] * len(dens2)
            for i in idx_thr2:
                cols2[i] = cur_pal[1]

            width = .8
            x = np.linspace(0, dens1.shape[0] - width, dens1.shape[0])
            fig, ax = plt.subplots(1, 2, sharey=True)
            ax[0].bar(x, dens1, color=cols1)
            ax[0].set_xlabel('stack height')
            ax[0].set_ylabel('density of non-zero data points', labelpad=20)
            ax[0].set_xticks([10, dens1.shape[0] - 10])
            ax[0].set_xticklabels(['IPL', 'GCL'])

            plt.locator_params(axis='y', nbins=4)

            ax[1].bar(x, dens2, color=cols2)
            ax[1].set_xlabel('stack height')
            ax[1].set_xticks([10, dens2.shape[0] - 10])
            ax[1].set_xticklabels(['IPL', 'GCL'])

            plt.locator_params(axis='y', nbins=4)

            fig.suptitle('Density profile\n' + str(exp_date) + ': ' + eye + ': ' + str(cell_id), fontsize=16)
            fig.tight_layout()
            fig.subplots_adjust(top=.88)

            return fig


@schema
class Recording(dj.Manual):
    definition="""
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
                        print('Threshold for differentiated signal was: ', d_sigma*alpha)

                        fig_v = self.show_spiketimes(voltage_trace, spiketimes, start, end, fs)
                        fig_dv = self.show_spiketimes(d_voltage, spiketimes, start, end, fs)

                        display(fig_v, fig_dv)

                        adjust1 = bool(int(input('Adjust threshold again? (Yes: 1, No: 0): ')))
                        plt.close(fig_v)
                        plt.close(fig_dv)



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
                 'lines.linewidth':2
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

            return  fig

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
    sta         : longblob	# spike-triggered average
    rf          : longblob  # array (stim_dim_x, stim_dim_y) with rf map at time point of kernel peak
    tau         : int       # time lag at which rf map is extracted
    kernel      : longblob  # time kernel from center pixel and its neighbours
    u           : longblob  # first temporal filter component
    s           : longblob  # singular values
    v           : longblob  # first spatial filter component
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
        stim_dim_x, stim_dim_y = (BWNoiseFrames() & key).fetch1['stim_dim_x','stim_dim_y']




        stimInd = np.zeros(rec_len).astype(int)-1

        if len(triggertimes) != stim_length:
            print('Something went wrong with the trigger detection\n # trigger: ', len(triggertimes))

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

        sta_raw = np.mean(ste,0)

        sta = scimage.filters.gaussian_filter(sta_raw.reshape(sta_raw.shape[0],stim_dim_x, stim_dim_y), [.2, .7, .7])

        # calculate time kernel from center pixel
        sd_map = np.std(sta, 0)
        idx_center = np.where(sd_map == np.max(sd_map))
        kernel = sta[:, idx_center[0], idx_center[1]]
        frame = np.where(abs(kernel) == abs(kernel).max())[0][0]
        rf = sta[frame, :, :]
        tau = int(100 - 10 * int(frame))

        try:

            (u, s, v) = np.linalg.svd(sta_raw)

        except Exception as e_svd:

            print(e_svd)
            u = np.zeros([sta_raw.shape[0], sta_raw.shape[0]]) # first temporal filter component
            v = np.zeros([sta_raw.shape[1], sta_raw.shape[1]]) # first spatial filter component

        if np.sign(np.mean(u[:, 0])) != np.sign(np.mean(kernel)):
            u = -1 * u

        if np.sign(np.mean(v[0, :])) != np.sign(np.mean(rf)):
            v = -1 * v

        self.insert1(dict(key,sta=sta_raw, rf = rf, tau = tau, kernel = kernel, u = u[:,0], s = np.diag(s), v = v[0,:]))

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

                return fig

    def plt_contour(self):

        from matplotlib import ticker

        plt.rcParams.update(
            {'figure.figsize': (12, 8),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .3,
             'lines.linewidth': 2
             }
        )

        for key in self.project().fetch.as_dict:
            rf = (self & key).fetch1['rf']
            tau = (self & key).fetch1['tau']
            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']


            fig = plt.figure()
            plt.title('$\Delta$ t: ' + str(tau) + '\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            im = plt.imshow(rf, interpolation='none',cmap=plt.cm.Greys_r, origin='upper')
            cs = plt.contour(rf,cmap=plt.cm.coolwarm, origin='upper', linewidth=4)

            cb = plt.colorbar(cs, extend='both', shrink=.8)
            cbaxes = fig.add_axes([.15, .02, .6, .03])  # [left, bottom, width, height]rf
            cbi = plt.colorbar(im, orientation='horizontal', cax=cbaxes)

            tick_locator = ticker.MaxNLocator(nbins=6)
            cbi.locator = tick_locator
            cbi.update_ticks()

            cb.locator = tick_locator
            cb.update_ticks()

            return fig

    def plt_svd(self):



        for key in self.project().fetch.as_dict:

            plt.rcParams.update(
                {'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'figure.figsize': (15, 8),
                 'figure.subplot.hspace': .2,
                 'figure.subplot.wspace': .1,
                 'ytick.major.pad': 10
                 }
            )

            cur_pal = sns.color_palette()

            rf = (self & key).fetch1['rf']
            tau = (self & key).fetch1['tau']
            kernel = (self & key).fetch1['kernel']
            u = (self & key).fetch1['u']
            v = (self & key).fetch1['v']

            stimDim = (BWNoiseFrames() & key).fetch1['stim_dim_x', 'stim_dim_y']

            fname = key['filename']
            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']

            fig = plt.figure()
            fig.suptitle(' STA at $\Delta$ t: ' + str(tau) + ' ms (upper panel) and SVD (lower panel) \n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

            fig.add_subplot(2, 3, 1)

            im = plt.imshow(rf, interpolation='none',cmap=plt.cm.coolwarm, origin='upper')
            cbi = plt.colorbar(im)
            plt.xticks([])
            plt.yticks([])
            tick_locator = ticker.MaxNLocator(nbins=6)
            cbi.locator = tick_locator
            cbi.update_ticks()

            fig.add_subplot(2, 2, 2)
            deltat = 1000  # in ms
            t = np.linspace(100, -deltat, len(kernel))
            if np.sign(np.mean(kernel)) == -1:
                plt.plot(t, kernel, color=cur_pal[0],linewidth=4)
            else:
                plt.plot(t, kernel, color=cur_pal[2],linewidth=4)

            plt.locator_params(axis='y', nbins=4)
            ax = fig.gca()
            ax.set_xticklabels([])
            ax.set_xlim([100, -deltat])
            plt.ylabel('stimulus intensity', labelpad=20)

            fig.add_subplot(2, 3, 4)
            im = plt.imshow(v.reshape(stimDim[0], stimDim[1]), interpolation='none',cmap=plt.cm.coolwarm, origin='upper')
            cbi = plt.colorbar(im)
            plt.xticks([])
            plt.yticks([])
            tick_locator = ticker.MaxNLocator(nbins=6)
            cbi.locator = tick_locator
            cbi.update_ticks()
            plt.xticks([])
            plt.yticks([])

            fig.add_subplot(2, 2, 4)

            if np.sign(np.mean(u)) == -1:
                plt.plot(t, u, color='b',linewidth=4)
            else:
                plt.plot(t, u, color='r',linewidth=4)

            plt.locator_params(axis='y', nbins=4)
            ax = fig.gca()
            ax.set_xlim([100, -deltat])
            plt.xlabel('time [ms]', labelpad=10)
            plt.ylabel('stimulus intensity', labelpad=20)

            plt.subplots_adjust(top=.8)

            return fig

@schema
class Overlay(dj.Computed):

    definition="""
    # Overlay of linestack and receptive field map

    -> Morph
    -> STA
    ---
    morph_pad       :longblob   # morphology padded to rf map size
    morph_shift     :longblob   # morphology with com shifted onto rf center
    rf_pad          :longblob   # receptive field map upsampled to morph resolution
    idx_center      :blob       # tuple with rf center indices
    idx_soma        :blob       # tuple with com of padded morph
    shift_x         :int        # number of pixels shifted in x dim
    shift_y         :int        # number of pixels shifted in y dim

    """

    def _make_tuples(self,key):

        stack = (Cut() & key).fetch1['stack_wos']
        (scan_z, scan_x, scan_y) = (Morph() & key).fetch1['scan_z', 'scan_x', 'scan_y']
        zoom = (Morph() & key).fetch1['zoom']
        scan_size = (Morph() & key).fetch1['scan_size']
        (dx_morph, dy_morph) = (Morph() & key).fetch1['dx','dy']
        (dx, dy) = (BWNoise() & key).fetch1['delx', 'dely']

        rf = (STA() & key).fetch1['rf']


        morph = np.mean(stack, 0)

        dely = (rf.shape[1] * dy - scan_size) / 2  # missing at each side of stack to fill stimulus in um
        delx = (rf.shape[0] * dx - scan_size) / 2

        ny_pad = int(dely / dy_morph)  # number of pixels needed to fill the gap
        nx_pad = int(delx / dx_morph)

        morph_pad = np.lib.pad(morph, ((nx_pad, nx_pad), (ny_pad, ny_pad)), 'constant', constant_values=0)

        factor = (morph_pad.shape[0] / rf.shape[0], morph_pad.shape[1] / rf.shape[1])

        rf_pad = scimage.zoom(rf, factor, order=0)

        params_rf = scimage.extrema(rf_pad)
        (off, on, off_ix, on_ix) = params_rf

        if abs(off) > abs(on):
            idx_rfcenter = off_ix
        else:
            idx_rfcenter = on_ix

        (idx_soma, idy_soma) = scimage.center_of_mass(morph_pad)

        shift_x = int(idx_rfcenter[0] + factor[0] / 2 - int(idx_soma))
        shift_y = int(idx_rfcenter[1] + factor[1] / 2 - int(idy_soma))

        try:
            morph_shift = np.lib.pad(morph, (
                (nx_pad + int(shift_x), nx_pad - int(shift_x)), (ny_pad + int(shift_y), ny_pad - int(shift_y))),
                                     'constant',
                                     constant_values=0)
        except Exception as e1:
            print(e1)
            print('RF borders reached')

            if abs(shift_x) > nx_pad:

                if abs(shift_y) < ny_pad:

                    if shift_x > nx_pad:

                        morph_shift = np.lib.pad(morph, (
                            (int(2 * nx_pad), 0), (ny_pad + int(shift_y), ny_pad - int(shift_y))),
                                                 'constant',
                                                 constant_values=0)
                    else:
                        morph_shift = np.lib.pad(morph, (
                            (0, int(2 * nx_pad)), (ny_pad + int(shift_y), ny_pad - int(shift_y))),
                                                 'constant',
                                                 constant_values=0)
                elif abs(shift_y) > ny_pad:

                    if (shift_x > nx_pad) & (shift_y > ny_pad):

                        morph_shift = np.lib.pad(morph, (
                            (int(2*nx_pad),0), (int(2 * ny_pad), 0)),
                                                 'constant',
                                                 constant_values=0)
                    elif (shift_x < nx_pad) & (shift_y > ny_pad):
                        morph_shift = np.lib.pad(morph, (
                            (0,int(2*nx_pad)), (int(2 * ny_pad),0)),
                                                 'constant',
                                                 constant_values=0)
                    elif (shift_x > nx_pad) & (shift_y < ny_pad):
                        morph_shift = np.lib.pad(morph, (
                            (int(2 * nx_pad),0), (0,int(2 * ny_pad))),
                                                 'constant',
                                                 constant_values=0)
                    elif (shift_x < nx_pad) & (shift_y < ny_pad):
                        morph_shift = np.lib.pad(morph, (
                            (0,int(2 * nx_pad)), (0, int(2 * ny_pad))),
                                                 'constant',
                                                 constant_values=0)


            elif abs(shift_y) > ny_pad:

                    if shift_y > ny_pad:

                        morph_shift = np.lib.pad(morph, (
                            (nx_pad + int(shift_x), nx_pad - int(shift_x)), (int(2 * ny_pad), 0)),
                                                 'constant',
                                                 constant_values=0)
                    else:
                        morph_shift = np.lib.pad(morph, (
                            (nx_pad + int(shift_x), nx_pad - int(shift_x)), (0, int(2 * ny_pad))),
                                                 'constant',
                                                 constant_values=0)
        self.insert1(dict(key,
                          morph_pad = morph_pad,
                          morph_shift = morph_shift,
                          rf_pad = rf_pad,
                          idx_center = np.array([idx_rfcenter]),
                          idx_soma = np.array([int(idx_soma),int(idy_soma)]),
                          shift_x = int(shift_x),
                          shift_y = int(shift_y)
                          ))


    def gaussian(self,height, mu_x, mu_y, sd_x, sd_y):
            """Returns a gaussian function with the given parameters"""
            sd_x = float(sd_x)
            sd_y = float(sd_y)
            return lambda x, y: height * np.exp(-((x - mu_x) ** 2 / (sd_x ** 2) + (y - mu_y) ** 2 / (sd_y ** 2)) / 2)

    def moments(self,data):
        """Returns (height,mu_x, mu_y, sd_x, sd_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        mu_x = (X * data).sum() / total
        mu_y = (Y * data).sum() / total
        col = data[:, int(mu_y)]
        sd_x = np.sqrt(np.abs((np.arange(col.size) - mu_y) ** 2 * col / col.sum()).sum())
        row = data[int(mu_x), :]
        sd_y = np.sqrt(np.abs((np.arange(row.size) - mu_x) ** 2 * row / row.sum()).sum())
        height = data.max()
        return height, mu_x, mu_y, sd_x, sd_y

    def fitgaussian(self,data):
        """Returns (mu_x, mu_y, sd_x, sd_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = self.moments(data)
        errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape)) -
                                           data)
        p, success = scoptimize.leastsq(errorfunction, params)
        return p

    def overlay(self):

        for key in self.project().fetch.as_dict:

            plt.rcParams.update(
                {'figure.figsize': (12, 8),
                 'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'figure.subplot.hspace': .2,
                 'figure.subplot.wspace': .3
                 }
            )

            morph_pad = (self & key).fetch1['morph_pad']
            morph_shift = (self & key).fetch1['morph_shift']
            rf_pad = (self & key).fetch1['rf_pad']
            (shift_x, shift_y) = (self & key).fetch1['shift_x', 'shift_y']
            (dx_morph, dy_morph) = (Morph() & key).fetch1['dx', 'dy']

            exp_date = (Experiment() & key).fetch1['exp_date']
            eye = (Experiment() & key).fetch1['eye']
            cell_id = (Cell() & key).fetch1['cell_id']

            line_pad = np.ma.masked_where(morph_pad == 0, morph_pad)
            line_shift = np.ma.masked_where(morph_shift == 0, morph_shift)

            clim = (0,.1)

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(rf_pad, cmap=plt.cm.coolwarm)
            ax[0].imshow(line_pad, cmap=plt.cm.gray, clim=clim)

            ax[0].set_xticklabels([])
            ax[0].set_yticklabels([])
            ax[0].set_title('original')

            ax[1].imshow(rf_pad, cmap=plt.cm.coolwarm)
            ax[1].imshow(line_shift, cmap=plt.cm.gray, clim=clim)

            dx_mu = shift_x * dx_morph
            dy_mu = shift_y * dy_morph

            ax[1].set_xticklabels([])
            ax[1].set_yticklabels([])
            ax[1].set_title('shifted by (%.1f , %.1f) $\mu m$' % (dx_mu, dy_mu),)

            plt.suptitle('Overlay rf and morph\n' + str(exp_date) + ': ' + eye + ': ' + str(cell_id),fontsize=16)

            plt.tight_layout()
            plt.subplots_adjust(top=.8)

            return fig

    # def overlay_gauss(self):
    #
    #     for key in self.project().fetch.as_dict:
    #
    #         plt.rcParams.update(
    #             {'figure.figsize': (12, 8),
    #              'axes.titlesize': 16,
    #              'axes.labelsize': 16,
    #              'xtick.labelsize': 16,
    #              'ytick.labelsize': 16,
    #              'figure.subplot.hspace': .2,
    #              'figure.subplot.wspace': .3,
    #              'lines.linewidth':1
    #              }
    #         )
    #
    #         morph_pad = (self & key).fetch1['stack_pad']
    #         morph_shift = (self & key).fetch1['stack_shift']
    #         rf_pad = (self & key).fetch1['rf_pad']
    #         (shift_x, shift_y) = (self & key).fetch1['shift_x', 'shift_y']
    #         params_m, params_m_shift, params_rf = (self & key).fetch1['gauss_m','gauss_m_shift','gauss_rf']
    #
    #         (dx_morph, dy_morph) = (Morph() & key).fetch1['dx', 'dy']
    #         (df_size_x,df_size_y) = (Morph() & key).fetch1['df_size_x','df_size_y']
    #
    #         exp_date = (Experiment() & key).fetch1['exp_date']
    #         eye = (Experiment() & key).fetch1['eye']
    #         cell_id = (Cell() & key).fetch1['cell_id']
    #
    #         rf_size_x = 2*params_rf[3]*dx_morph
    #         rf_size_y = 2*params_rf[4]*dy_morph
    #
    #         line_pad = np.ma.masked_where(morph_pad == 0, morph_pad)
    #         line_shift = np.ma.masked_where(morph_shift == 0, morph_shift)
    #
    #         fit_m_pad = self.gaussian(*params_m)
    #         fit_rf_pad = self.gaussian(*params_rf)
    #         fit_m_shift = self.gaussian(*params_m_shift)
    #
    #         clim = (0,1)
    #
    #         fig, ax = plt.subplots(1, 2)
    #         ax[0].imshow(rf_pad, cmap=plt.cm.coolwarm)
    #         ax[0].imshow(line_pad, cmap=plt.cm.gray, clim=clim)
    #         ax[0].contour(fit_m_pad(*np.indices(morph_pad.shape)), cmap=plt.cm.Greens, linewidth=1)
    #         ax[0].contour(fit_rf_pad(*np.indices(rf_pad.shape)), cmap=plt.cm.Purples, linewidth=1)
    #
    #         ax[0].set_xticklabels([])
    #         ax[0].set_yticklabels([])
    #         ax[0].set_title('original')
    #         ax[0].annotate('rf 2 s.d. in x [um]: %.2f \nrf 2 s.d. in y [um]: %.2f' % (rf_size_x, rf_size_y),
    #                           xy=(20, 150), fontsize=14)
    #
    #
    #         ax[1].imshow(rf_pad, cmap=plt.cm.coolwarm)
    #         ax[1].imshow(line_shift, cmap=plt.cm.gray, clim=clim)
    #         ax[1].contour(fit_m_shift(*np.indices(morph_pad.shape)), cmap=plt.cm.Greens, linewidth=1)
    #         ax[1].contour(fit_rf_pad(*np.indices(rf_pad.shape)), cmap=plt.cm.Purples, linewidth=1)
    #
    #         dx_mu = shift_x * dx_morph
    #         dy_mu = shift_y * dy_morph
    #
    #         ax[1].set_xticklabels([])
    #         ax[1].set_yticklabels([])
    #         ax[1].set_title('shifted by (%.1f , %.1f) $\mu m$' % (dx_mu, dy_mu))
    #         ax[1].annotate('df size in x [um]: %.2f\ndf size in y [um]: %.2f' % (df_size_x, df_size_y), xy=(20, 150),
    #                        fontsize=14)
    #
    #         plt.suptitle('Overlay rf and morph\n' + str(exp_date) + ': ' + eye + ': ' + str(cell_id), fontsize=16)
    #
    #         plt.tight_layout()
    #         plt.subplots_adjust(top=.8)
    #
    #         return fig

    def center(self):

        for key in self.project().fetch.as_dict:

            plt.rcParams.update(
                {'figure.figsize': (12, 8),
                 'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'figure.subplot.hspace': .2,
                 'figure.subplot.wspace': .3
                 }
            )

            rf_pad = (self & key).fetch1['rf_pad']
            morph_pad = (self & key).fetch1['morph_pad']
            idx_rfcenter = (self & key).fetch1['idx_center']
            idx_soma = (self & key).fetch1['idx_soma']
            stimDim = (BWNoiseFrames() & key).fetch1['stim_dim_x','stim_dim_y']

            factor = (morph_pad.shape[0]/stimDim[0],morph_pad.shape[1]/stimDim[1])

            fig, ax = plt.subplots(1, 2)

            ax[0].imshow(rf_pad, cmap=plt.cm.coolwarm)
            ax[0].scatter(idx_rfcenter[0][1] + int(factor[1] / 2), idx_rfcenter[0][0] + int(factor[0] / 2), marker='x', s=100,
                          linewidth=3, color='k', label='max|w|')

            ax[0].set_xticklabels([])
            ax[0].set_yticklabels([])
            ax[0].set_title('Receptive Field')
            ax[0].legend()

            ax[1].imshow(morph_pad, clim=(0, .01))
            ax[1].scatter(idx_soma[1], idx_soma[0], marker='x', s=100, linewidth=3, color='b', label='com')

            ax[1].set_xticklabels([])
            ax[1].set_yticklabels([])
            ax[1].set_title('Dendritic Field')
            ax[1].legend()

            return fig


@schema
class Blur(dj.Computed):

    definition="""
    -> Overlay
    ---
    min_res         : double    # minimum total residual sum, per pixel
    sig_minres      : double    # blurring filter size with which min residual was observed
    max_r           : double    # maximum correlation with rf
    sig_maxr        : double    # blurring filter size with which max correlation was observed
    rf_z            : longblob  # normalized rf
    df_z_minres     : longblob  # normalized, blurred df
    df_z_maxr       : longblob  # normalized, blurred df
    """


    def _make_tuples(self,key):

        stack = (Cut() & key).fetch1['stack_wos']
        (scan_z, scan_x, scan_y) = (Morph() & key).fetch1['scan_z', 'scan_x', 'scan_y']
        zoom = (Morph() & key).fetch1['zoom']
        scan_size = (Morph() & key).fetch1['scan_size']
        (dx_morph, dy_morph) = (Morph() & key).fetch1['dx', 'dy']

        shift_x, shift_y = (Overlay() & key).fetch1['shift_x', 'shift_y']
        (dx, dy) = (BWNoise() & key).fetch1['delx', 'dely']
        rf = (STA() & key).fetch1['rf']

        morph = np.mean(stack, 0)

        dely = (rf.shape[1] * dy - scan_size) / 2  # missing at each side of stack to fill stimulus in um
        delx = (rf.shape[0] * dx - scan_size) / 2

        ny_pad = int(dely / dy_morph)  # number of pixels needed to fill the gap
        nx_pad = int(delx / dx_morph)

        morph_pad = np.lib.pad(morph, ((nx_pad + shift_x, nx_pad - shift_x), (ny_pad + shift_y, ny_pad - shift_y)),
                               'constant', constant_values=0)

        # Normalize RF

        b = np.mean(rf[0, :])
        peak = abs(rf).max()
        rf_z = (rf - b) / peak

        # Blur
        sig = np.arange(0, 100, 10)
        blur = {}
        blur.clear()
        blur['blur'] = []
        blur['sigma'] = []

        for s in sig:
            blur['blur'].append(scimage.gaussian_filter(morph_pad, sigma=s))  # blur
            blur['sigma'].append(s)

        blur_df = pd.DataFrame(blur)

        df_nz = []
        for ix, row in blur_df.iterrows():
            df_nz.append(scmisc.imresize(row.blur, rf.shape, interp='bicubic'))  # down-sample

        blur_df = blur_df.assign(df_nz=df_nz)

        df_z = []
        res = []
        res_n = []
        res_sum = []
        r = []

        for ix, row in blur_df.iterrows():
            if abs(rf_z.min()) > abs(rf_z.max()):
                df_z.append(-(row.df_nz / abs(row.df_nz).max()))
            else:
                df_z.append((row.df_nz / abs(row.df_nz).max()))
            res.append(df_z[ix] - rf_z)
            res_n.append(res[ix] / abs(res[ix]).max())
            res_sum.append(abs(res[ix]).sum() / len(rf.flatten()))
            r.append(np.corrcoef(df_z[ix].flatten(), rf_z.flatten())[0, 1])

        blur_df = blur_df.assign(df=df_z)
        blur_df = blur_df.assign(res=res)
        blur_df = blur_df.assign(res_n=res_n)
        blur_df = blur_df.assign(res_sum=res_sum)
        blur_df = blur_df.assign(r=r)

        ix_min_res = blur_df.res_sum.idxmin()
        ix_max_r = blur_df.r.idxmax()

        self.insert1(dict(key,
                          min_res = blur_df.res_sum.min(),
                          sig_minres = blur_df.sigma[ix_min_res],
                          sig_maxr = blur_df.sigma[ix_max_r],
                          max_r = blur_df.r.max(),
                          rf_z = rf_z,
                          df_z_minres = blur_df.df[ix_min_res],
                          df_z_maxr = blur_df.df[ix_max_r]
                          ))

    def plt_minres(self):

        for key in self.project().fetch.as_dict:

            plt.rcParams.update(
                {'figure.figsize': (20, 10),
                 'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'figure.subplot.hspace': 0.1,
                 'figure.subplot.wspace': 0.2
                 }
            )

            df_minres = (self & key).fetch1['df_z_minres']
            df_maxr = (self & key).fetch1['df_z_maxr']
            rf_z = (self & key).fetch1['rf_z']
            minres = (self & key).fetch1['min_res']
            maxr = (self & key).fetch1['max_r']
            sig_minres = (self & key).fetch1['sig_minres']
            sig_maxr = (self & key).fetch1['sig_maxr']

            rf_pad = (Overlay() & key).fetch1['rf_pad']
            stack_pad = (Overlay() & key).fetch1['morph_shift']

            blur = scimage.gaussian_filter(stack_pad, sigma=.7)
            line_bl = np.ma.masked_where(blur == 0, blur)


            fig = plt.figure()
            gs1 = gridsp.GridSpec(2, 1)
            gs1.update(right=.48)
            ax = plt.subplot(gs1[:, :])
            im = ax.imshow((rf_pad - rf_pad[0, :].mean()) / abs(rf_pad).max(), cmap=plt.cm.coolwarm,
                           interpolation='nearest')
            li = ax.imshow(line_bl, cmap=plt.cm.Greys_r)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            cbar = plt.colorbar(im, ax=ax, format='%.1f', shrink=.9)
            cbar.set_label('normed rf', labelpad=40, rotation=270)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()

            gs2 = gridsp.GridSpec(2, 2)
            gs2.update(left=.55, right=.98)
            ax = plt.subplot(gs2[0, 0])
            im = ax.imshow(df_minres, cmap=plt.cm.coolwarm, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('$\sigma$ : %.1f' % (sig_minres))
            cbar = plt.colorbar(im, ax=ax, format='%.1f', shrink=.95)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()

            ax = plt.subplot(gs2[0, 1])
            im = ax.imshow(df_minres - rf_z, cmap=plt.cm.coolwarm, clim=(-1, 1), interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('$\Sigma|rf-df_\sigma|_{min}$ : %.2f' % (minres))
            cbar = plt.colorbar(im, ax=ax, format='%.1f', shrink=.95)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()

            ax = plt.subplot(gs2[1, 0])
            im = ax.imshow(df_maxr, cmap=plt.cm.coolwarm, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('$\sigma$ : %.1f' % (sig_maxr))
            cbar = plt.colorbar(im, ax=ax, format='%.1f', shrink=.95)
            cbar.set_label('blurred df', labelpad=20, rotation=270, y=1.1)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()

            ax = plt.subplot(gs2[1, 1])
            im = ax.imshow(df_maxr - rf_z, cmap=plt.cm.coolwarm, clim=(-1, 1), interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('$corr_{max}$ : %.2f' % (maxr))
            cbar = plt.colorbar(im, ax=ax, format='%.1f', shrink=.95)
            cbar.set_label('residual', labelpad=30, rotation=270, y=1.1)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()

            fig.suptitle(str(key['exp_date']) + ': ' +  key['eye'] + ':' + str(key['cell_id']) + ': ' + key['filename'], fontsize=18)
            fig.subplots_adjust(top=.88)


            return fig


# class LnpExp(dj.Computed):
#
#     definition="""
#     # Fit a RF under a LNP model with exponential non-linearity
#     -> STA
#     ---
#     frames_conv : longblob   # stimulus frames convolved with time kernel in steps of 200 ms
#     sta_inst    : longblob   # instantaneous rf obtained by convolution of STA with the time kernel from svd
#     rf          : longblob   # fited rf with best negative log-likelihood on the test set
#     pred        : double     # percentage of time bins where number of spikes was predicted correctly on test set
#     pears_r     : double     # Pearson's correlation coefficient between predicted and true psth on test set
#     r2          : double     # fraction variance explained on test set
#     pred_psth   : longblob   # array (1 x T_test) predicted psth with time bins of 200ms on the test set
#     true_psth   : longblob   # array (1 x T_test) spiketimes vector binned by 200 ms with length of test set
#
#     """
#
#     def _make_tuples(self,key):
#
#         u = (STA() & key).fetch1['u']
#
#         stimDim = (BWNoiseFrames() & key).fetch1['stim_dim_x', 'stim_dim_y']
#         Frames = (BWNoiseFrames() & key).fetch1['frames']
#
#         freq = (BWNoise() & key).fetch1['freq']
#
#         fs = (Recording() & key).fetch1['fs']
#
#         spiketimes = (Spikes() & key).fetch1['spiketimes']
#         rec_len = (Spikes() & key).fetch1['rec_len']
#         triggertimes = (Trigger() & key).fetch1['triggertimes']
#
#
#
#         k = u[::20]  # in time steps of 200 ms which corresponds to stimulation frequency of 5 Hz
#         k_pad = np.vstack(
#             (np.zeros(k[:, None].shape), k[:, None]))  # zero-padding to shift origin of weights vector accordingly
#
#         s_conv = scimage.filters.convolve(Frames, k_pad)
#
#         stimInd = np.zeros([rec_len, 1]).astype(int) - 1
#
#         for n in range(len(triggertimes) - 1):
#             stimInd[triggertimes[n]:triggertimes[n + 1] - 1] += int(n + 1)
#         stimInd[triggertimes[len(triggertimes) - 1]:triggertimes[len(triggertimes) - 1] + (fs / freq)] += int(
#             len(triggertimes))
#
#         spiketimes = spiketimes[spiketimes > triggertimes[0]]
#         spiketimes = spiketimes[spiketimes < triggertimes[len(triggertimes) - 1] + (fs / freq)]
#         nspiketimes = len(spiketimes)
#
#         ste = np.zeros([nspiketimes, stimDim[0] * stimDim[1]])
#
#         for s in range(nspiketimes):
#             ste[s, :] = s_conv[stimInd[spiketimes[s]], :]
#
#         sta_inst = np.mean(ste, 0)
#
#         s = np.transpose(s_conv)  # make it a (n x T) array
#         T = s.shape[1]
#
#         # bin spiketimes in 200ms time bins
#         y = np.histogram(spiketimes, bins=T,
#                          range=[triggertimes[0], triggertimes[len(triggertimes) - 1] + (fs / freq)])[0]
#         LNP_dict = {}
#         LNP_dict.clear()
#         LNP_dict['nLL train'] = []
#         LNP_dict['nLL test'] = []
#         LNP_dict['w'] = []
#         LNP_dict['pred correct'] = []
#         LNP_dict['pearson r'] = []
#         LNP_dict['R2'] = []
#         LNP_dict['pred psth'] = []
#         LNP_dict['true psth'] = []
#
#         w0 = np.zeros(s.shape[0])
#         k= 6
#         kf = KFold(T, n_folds=k)
#
#         for train, test in kf:
#             res = scoptimize.minimize(self.nll_exp, w0, args=(s[:, train], y[train]), jac=True, method='TNC')
#             print(res.message, 'neg log-liklhd: ', res.fun)
#
#             LNP_dict['nLL train'].append(res.fun)
#             LNP_dict['nLL test'].append(self.nll_exp(res.x, s[:, test], y[test])[0])
#             LNP_dict['w'].append(res.x)
#
#             y_test = np.zeros(len(test))
#             for t in range(len(test)):
#                 r = np.exp(np.dot(res.x, s[:, test[t]]))
#                 y_test[t] = np.random.poisson(lam=r)
#
#             LNP_dict['pred correct'].append((sum(y_test == y[test]) / len(test)))
#             LNP_dict['pearson r'].append(scstats.pearsonr(y_test, y[test])[0])
#             LNP_dict['R2'].append(1 - np.sum(np.square(y[test] - y_test)) / np.sum(np.square(y[test] - np.mean(y[test]))))
#             LNP_dict['pred psth'].append(y_test * freq)
#             LNP_dict['true psth'].append(y[test] * freq)
#         LNP_df = pd.DataFrame(LNP_dict)
#
#         idx = LNP_df['nLL test'].idxmin()
#         print('idx: ', idx)
#         print(LNP_df['w'][idx].shape)
#         print(LNP_df['true psth'].shape)
#
#         self.insert1(dict(key,frames_conv = s_conv,
#                           sta_inst = sta_inst,
#                           rf = LNP_df['w'][idx],
#                           pred = LNP_df['pred correct'][idx],
#                           pears_r = LNP_df['pearson r'][idx],
#                           r2  = LNP_df['R2'][idx],
#                           pred_psth = LNP_df['pred psth'][idx],
#                           true_psth = LNP_df['true psth'][idx]
#                           ))
#
#     def nll_exp(self,wT,s,y):
#
#         """
#             Compute the negative log-likelihood of an LNP model wih exponential non-linearity
#
#             :arg wT: current receptive field array(stimDim[0]*stimDim[1],)
#             :arg s: stimulus array(stimDim[0]*stimDim[1],T)
#             :arg y: spiketimes array(,T)
#
#             :return nLL: computed negative log-likelihood scalar
#             :return dnLL: computed first derivative of the nLL
#             """
#
#         r = np.exp(np.dot(wT, s))
#         nLL = np.dot(r - y * np.log(r), np.ones(y.shape))
#
#         dnLL = np.dot(s * r - y * s, np.ones(y.shape))
#
#         return nLL, dnLL


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
        plt.rcParams.update(
            {'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.figsize': (12, 8),
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2,
             'lines.linewidth':2,
             'ytick.major.pad': 10})

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

            plt.tight_layout()
            plt.subplots_adjust(top=.8)

            return fig


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

        plt.rcParams.update(
            {'axes.titlesize': 16,
             'axes.labelsize': 16,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'figure.figsize': (10, 8),
             'figure.subplot.hspace': .2,
             'figure.subplot.wspace': .2,
             'lines.linewidth': 2,
             'ytick.major.pad': 10})

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
                plt.suptitle('Directional Tuning\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

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

                plt.tight_layout()
                plt.subplots_adjust(top=.8)

                return fig

    def plt_ds_traces(self):

        for key in self.project().fetch.as_dict:

            plt.rcParams.update(
                {'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'figure.figsize': (20, 8),
                 'figure.subplot.hspace': .1,
                 'figure.subplot.wspace': .2,
                 'lines.linewidth': 2,
                 'ytick.major.pad': 10
                 }
            )

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

                plt.tight_layout()
                plt.subplots_adjust(top=.8)

                return fig1, fig2

            plt.tight_layout()
            plt.subplots_adjust(top=.8)

            return fig1


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
            spikes_norm.append(spiketimes[(spiketimes > tr) & (spiketimes <  tr + 2*t_off + 2*t_on)])
            spikes_off.append(spiketimes[(spiketimes > tr) & (spiketimes < tr + t_off)])
            spikes_on.append(spiketimes[(spiketimes > tr + t_off) & (spiketimes < tr + t_off + 2 * t_on)])


        pol = len(spikes_on) / len(spikes_off)
        self.insert1(dict(key, pol=pol))



    def plt_on_off(self):

        for key in self.project().fetch.as_dict:

            plt.rcParams.update(
                {'axes.titlesize': 16,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 16,
                 'ytick.labelsize': 16,
                 'figure.figsize': (15, 8),
                 'figure.subplot.hspace': .2,
                 'figure.subplot.wspace': .2,
                 'lines.linewidth': 2,
                 'ytick.major.pad': 10
                 }
            )

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

            fig1, axarr = plt.subplots(4, int(np.ceil(len(triggertimes) / 2)), sharex=True, sharey=True)
            plt.suptitle('Spot response\n' + str(exp_date) + ': ' + eye + ': ' + fname, fontsize=16)

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

                plt.tight_layout()
                plt.subplots_adjust(top=.8)
                return fig1, fig2

            plt.tight_layout()
            plt.subplots_adjust(top=.8)
            return fig1


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
