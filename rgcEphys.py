import datajoint as dj
import h5py
import numpy as np

schema = dj.schema('rgcEphys',locals())	# decorator for all classes that represent tables from the database 'rgcEphys'

@schema
class Animal(dj.Manual):
	definition = """
	# Basic animal info
	
	animal_id			:varchar(20   																# unique ID given to the animal
	---
	species="mouse"		:enum("mouse","rat","zebrafish")											# animal species
	animal_line			:enum("PvCreAi9","B1/6","ChATCre","PvCreTdT","PCP2TdT","ChATCreTdT","WT")	# transgnenetic animal line, here listed: mouse lines
	gender="unknown"	:enum("M","F","unknown")													# gender
	date_of_birth		:date																		# date of birth
	"""


@schema
class Experiment(dj.Manual):
	definition = """
	# Basic experiment info
	
	-> Animal
	
	exp_date	:date										# date of recording
	eye			:enum("R","L")								# left or right eye of the animal
	---
	experimenter				:varchar(20)				# first letter of first name + last name = lrogerson/tstadler
	setup="1"					:tinyint unsigned			# setup 1-3
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
	# PUT MORE INFO HERE
	"""

	
@schema
class Recording(dj.Manual):
	definition="""
	# Stimulus information for a particular recording
	
	->Cell
	
    filename		:varchar(200) 							# name of the converted recording file
    ---
    stim_type		:enum("bw_noise","chirp","ds","on_off")	# type of stimulus played during the recording
	"""

@schema
class Rawdata(dj.Imported):
	definition="""
	# Rawdata extracted from h5 file
	
	->Recording
	---
	rawtrace		:longblob	# array containing the raw voltage trace
	triggertrace	:longblob	# array containing the light trigger trace
	"""
	
	def _make_tuples(self,key):
		# fetch required data
		fname = (Recording() & key).fetch1['filename']
		cell_path = (Cell() & key).fetch1['folder']
		exp_path = (Experiment() & key).fetch1['path']
		
		# extract raw data for the given recording
		full_path = exp_path + '/' + cell_path + '/' + fname + '.h5'

        with h5py.File(full_path,'r') as f:
    		ch_keylist = [key for key in f['channels'].keys()]
		
