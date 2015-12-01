import datajoint as dj
import h5py
import numpy as np

schema = dj.schema('rgcEphys',locals())	# decorator for all classes that represent tables from the database 'rgcEphys'

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
	"""

@schema
class Morphology(dj.Manual):
	definition = """
	-> Experiment
	---
	"""

@schema
class Recording(dj.Manual):
	definition="""
	# Information for a particular recording file
	
	-> Cell
	filename		: varchar(200) 		  						# name of the converted recording file
    ---
    fs=10000		: int					  					# sampling rate of the recording
    stim_type		: enum('bw_noise','chirp','ds','on_off')	# type of the stimulus played during recording

	"""

@schema
class Spikes(dj.Computed):
	definition="""
	# Spike times in the Recording
	
	-> Recording
	---
	spiketimes_n		: longblob				# array spiketimes (1,nSpikes) containing spiketimes  in sample points
	spiketimes_ms		: longblob				# array spiketimes (1,nSpikes) containing spiketimes in ms
	"""
	
	def _make_tuples(self,key):
		# fetch required data
		fname, samp_rate = (Recording() & key).fetch1['filename','fs']
		cell_path = (Cell() & key).fetch1['folder']
		exp_path = (Experiment() & key).fetch1['path']
		
		# extract raw data for the given recording
		full_path = exp_path + '/' + cell_path + '/' + fname + '.h5'
		f = h5py.File(full_path,'r')
				
		ch_keylist = [key for key in f['channels'].keys()]
		
		rawdata = {}
		
		for ch in range(0,len(ch_keylist)):
			name_ch = f['channels'][ch_keylist[ch]][0].astype(str)  # 'ch{}_data'.format(ch)
			ch_grp = f[name_ch] # get each channel group into hdf5 grp object
			keylist = [key for key in ch_grp.keys()] # get key within one group
			rawdata[name_ch]  = ch_grp[keylist[1]]['data'][:] # initialize as section_00
			for sec in range(2,len(keylist)):
				ch_sec_tmp = ch_grp[keylist[sec]]
				dset = ch_sec_tmp['data'][:] # get array
				rawdata[name_ch] = np.append(rawdata[name_ch],dset)
		
		# determine threshold
		sigma = np.median(np.abs(rawdata['Vm_scaled AI #10'])/.6745)
		thr = 5 * sigma
		# print('Threshold is -', thr, 'mV')
		
		# threshold signal
		tmp = np.array(rawdata['Vm_scaled AI #10'])
		thr_boolean = [tmp > -thr]
		tmp[thr_boolean] = 0
		
		# detect spikes as threshold crossings
		tmp[tmp!=0]=1
		tmp = tmp.astype(int)
		tmp2 = np.append(tmp[1:len(tmp)],np.array([0],int))
		dif = tmp2-tmp
		
		s_n = np.where(dif==-1) # spiketimes in sample
		s_n = np.array(s_n)
		
		dt = 1/samp_rate*1000 # delta t in ms
		s_ms = s_n*dt # spiketimes in ms
		
		# submit
		self.insert1(dict(key, spiketimes_n=s_n, spiketimes_ms=s_ms))

@schema
class BWNoiseParams(dj.Lookup):
	definition="""
	noise_id	:int	# unique stimulus id for a set of parameter
	---
	freq	:int	# frequency in Herz
	delx	:int	# size of a pixel in x dimensions
	dely	:int	# size of a pixel in y dimensions
	mseq	:enum('BWNoise.txt','BWNoise_long.txt')	# name of mseq
	"""
	contents = [(0,5,40,40,'BWNoise.txt'),(1,20,40,40,'BWNoise.txt'),(2,5,20,20,'BWNoise.txt'),(3,20,20,20,'BWNoise.txt')]
@schema
class BWNoiseFrames(dj.Computed):
	definition="""
	# Stimulus frames for a given mseq
	-> BWNoiseParams
	---
	stim	:longblob	# array of stimulus frames
	"""
	def _make_tuples(self, key):
		# fetch required data

		file = (BWNoiseParams() & key).fetch1['mseq']
		stim_folder = '/notebooks/Data/Stadler/'

		# read stimulus information into np.array
		lines = open(stim_folder + file).read().split('\n')
		stimDim = np.asarray(lines[0].split(',')).astype(int)
		stim = np.asarray(lines[1].split(','),).astype(int).reshape(1,stimDim[0],stimDim[1])
		for l in range(2,len(lines)-1):
			stim = np.append(stim,np.asarray(lines[l].split(',')).astype(int).reshape(1,stimDim[0],stimDim[1]),axis=0)

		# insert data
		self.insert1(dict(key,stim=stim))

@schema
class BWNoise(dj.Computed):
	definition="""
	# Fetch parameter set and stimulus array from lookup table
	-> Recording
	-> BWNoiseParams
	# repeat_idx	:int auto_increment		# number of stimulus repeat with this parameter settings
	---
	params			:blob					# set of parameters for the noise stimulus played during the recording
	mseq			:varchar(50)			# name of the mseq txt file
	"""
	@property
	def populated_from(self):
		return Recording() & dict(stim_type='bw_noise')

	def _make_tuples(self, key):
		#fetch required data

		rec = (Recording() & key).fetch1['filename']
		cell_path = (Cell() & key).fetch1['folder']
		exp_path = (Experiment() & key).fetch1['path']

		# extract stim_id for the given recording
		header = exp_path + '/' + cell_path + '/' + 'header.txt'

		with open(header,'r') as f:
			for line in f:
				if rec in line:
					stim_id = int(line.split(',')[1])
					params = np.asarray((BWNoiseParams() & dict(noise_id=stim_id)).fetch1['freq','delx','dely'])
					mseq = (BWNoiseParams() & dict(noise_id=stim_id)).fetch1['mseq']
					self.insert1(dict(key,noise_id=stim_id,params=params,mseq=mseq))








