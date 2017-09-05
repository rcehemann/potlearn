#!/bin/python

import numpy  as np
import potdat as pd
import nav

class potList():
	def __init__(self, plist=[]):
		self.plist = plist

	def __len__(self):
		return len(self.plist)

	def __contains__(self, pot):
		return pot in self.plist
	
	def append(self, potData):
		self.plist.append(potData)

	def find_in_path(self, *argv):
		for i in range(0, len(argv)):
			path = str(*argv[i])
			paths = nav.find_all(path)
			if not paths:
				print "No valid potentials found!"
				exit()
			else:
				print "Found %i valid potentials in %s" % (len(paths), path)
			
			for path in paths:
				try:
					pot = pd.potData(path)
					self.append(pot)
				except:
					print "invalid potential " + path
					pass

	def get_prop_list(self, key, idx):
		return [pot.data[key][idx] for pot in self.plist if pot.data[key][idx] != ""]

	def get_prop_rms(self, key):
		rms = 0.0
		count = 0
		for pot in self.plist:
			err = pot.data[key][1]
			if err != "":
				count += 1
				rms += err**2
		return np.sqrt(rms/count)

	def get_type_int_list(self):
		return [pot.get_type_int() for pot in self.plist]

	def print_filenames(self):
		for pot in self.plist:
			print pot.filename

	def get_length(self):
		return len(self.plist)

	def get_err_lists(self):
		return [[pot.data[key][1] for key in pot.data] for pot in self.plist]
	
	def get_err_lists_subs(self, keys):
		return [[pot.data[key][1] for key in keys] for pot in self.plist]

	def get_rms_list(self):
		return [pot.get_rms_error() for pot in self.plist]

	def global_rms_error(self):
		errs = [pot.get_rms_error() for pot in self.plist]
		return np.mean(errs), np.std(errs)

	def get_max_rms(self):
		return max(self.get_rms_list())
	
	def get_max_rms_name(self):
		Max = self.get_max_rms()
		for pot in self.plist:
			if pot.get_rms_error() == Max:
				return pot.filename

	def trim_list_by_rms(self, max_rms):
		self.plist = [pot for pot in self.plist if pot.get_rms_error() <= float(max_rms)]

	def trim_list_by_empty(self):
		self.plist = [pot for pot in self.plist if '' not in np.r_[pot.data.values()]]

	def trim_list_by_empty_subs(self, keys):
		self.plist = [pot for pot in self.plist if '' not in [pot.data[key][1] for key in keys]]

	def trim_by_unique(self):
		self.plist = list(set(self.plist))

	def split_by_type(self):
		meamlist = potList([pot for pot in self.plist if pot.get_filename().split(".")[0] == "MEAM"])
		eamlist = potList([pot for pot in self.plist if pot.get_filename().split(".")[0] == "EAM"])
		return eamlist, meamlist
