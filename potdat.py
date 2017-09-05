#!/bin/python
import re
import numpy as np
class potData():
	def __init__(self):

		self.filename = ""
		self.data = {
				"dE_hcp": [],
				"dE_fcc": [],
				"dE_wti": [],
				"dE_bw" : [],
				"dE_bta": [],
				"a_bcc"	: [],
				"a_fcc" : [],
				"a_hcp" : [],
				"c_hcp"	: [],
				"a_bw"	: [],
				"a_bta" : [],
				"c_bta" : [],
				"a_wti" : [],
				"c_wti" : [],
				"C11"	: [],
				"C12"	: [],
				"C44"	: [],
				"vacfor": [],
				"vacmig": [],
				"vacact": [],
				"e100"  : [],
				"d100"  : [],
				"e110"  : [],
				"d110"  : [],
				"e111"  : [],
				"d111"  : [],
				"db100" : [],
				"db110" : [],
				"db111" : [],
				"crd"   : [],
				"oct"   : [],
				"tet"   : []
				}

	def __init__(self, filename):

		self.filename = filename.split("/")[-1]
		htmlkey = { 	
				"dE_hcp": "D<i>E</i><SUB> hcp-bcc</SUB>",
				"dE_fcc": "D<i>E</i><SUB> fcc-bcc</SUB>",
				"dE_wti": "D<i>E</i><SUB> &omega;Ti-bcc</SUB>",
				"dE_bw" : "D<i>E</i><SUB> &beta;W-bcc</SUB>",
				"dE_bta": "D<i>E</i><SUB>&beta;Ta-bcc</SUB>",
				"a_bcc"	: "<i>a</i> (A)",
				"a_fcc" : "<i>a</i><SUB> fcc</SUB>",
				"a_hcp" : "<i>a</i><SUB> hcp</SUB>",
				"c_hcp"	: "<i>c</i><SUB> hcp</SUB>",
				"a_bw"	: "<i>a</i><SUB> &beta;W</SUB>",
				"a_bta" : "<i>a</i><SUB> &beta;Ta</SUB>",
				"c_bta" : "<i>c</i><SUB> &beta;Ta</SUB>",
				"a_wti" : "<i>a</i><SUB> &omega;Ti</SUB>",
				"c_wti" : "<i>c</i><SUB> &omega;Ti</SUB>",
				"C11"	: "<i>C</i><SUB>11</SUB>",
				"C12"	: "<i>C</i><SUB>12</SUB>",
				"C44"	: "<i>C</i><SUB>44</SUB>",
				"vacfor": "Formation (eV)",
				"vacmig": "Migration (eV)",
				"vacact": "Activation energy (eV)",
				"e100"  : "{100} (meV/A<sup>2</sup>)",
				"d100"  : "D12 (%)",
				"e110"  : "{110} (meV/A<sup>2</sup>)",
				"d110"  : "D12 (%)",
				"e111"  : "{111} (meV/A<sup>2</sup>)",
				"d111"  : "D12 (%)",
				"db100" : "100</td>",
				"db110" : "110</td>",
				"db111" : "111</td>",
				"crd"   : "CRD</td>",
				"oct"   : "OCT</td>",
				"tet"   : "TET</td>"
				}

		f = open(filename, 'r')
		locations = {}
		for key in htmlkey:
			locations[key] = f.read().find(htmlkey[key])
			f.seek(0,0)
	
		f.seek(0,0)
		self.data = {}
		for key in htmlkey:

			# regex for fit error
			reger = "Total sum error is (.*?) \<\/h4\>"

			# regular expression to capture data from html fields
			regex = "\"\>(.*?)\</td\>"
			
			# another regex to extract elastic constants and bulk modulus for lower strain value
			regec = "\((.*?) : "			

			if locations[key] != -1:

				# exception for planar spacings because keys look the same
				if key == "d100":
					try:
						f.seek(locations["e100"])
						for i in range(7):
							f.readline()
	
						f.readline()
						potval = re.findall(regex, f.readline())
						f.readline()
					#	errval = re.findall(regex, f.readline())
						dftval = re.findall(regex, f.readline())
					except ValueError:
						raise("Couldn't find key " + key + " in file " + self.filename)
	
				elif key == "d110":
					try:
						f.seek(locations["e110"])
						for i in range(7):
							f.readline()
	
						f.readline()
						potval = re.findall(regex, f.readline())
						f.readline()
					#	errval = re.findall(regex, f.readline())
						dftval = re.findall(regex, f.readline())
					except ValueError:
						raise("Couldn't find key " + key + " in file " + self.filename)
	
				elif key == "d111":
					try:
						f.seek(locations["e111"])
						for i in range(7):
							f.readline()
	
						f.readline()
						potval = re.findall(regex, f.readline())
						f.readline()
				#		errval = re.findall(regex, f.readline())
						dftval = re.findall(regex, f.readline())
					except ValueError:
						raise("Couldn't find key " + key + " in file " + self.filename)
	
				else:	
	
					try:			
						f.seek(locations[key])
						if key == "dE_bta":
							f.readline()
						f.readline()
						potval = re.findall(regex, f.readline())
						f.readline()
				#		errval = re.findall(regex, f.readline())
						dftval = re.findall(regex, f.readline())
					except ValueError:
						raise("Couldn't find key " + key + " in file " + self.filename)
			
				if key.split()[0][0:2] == "C1":
					potval = re.findall(regec, potval[0])
		
				if not potval:
					potval = ""
				else:
					potval = potval[-1]
				if not dftval:
					dftval = ""
				else:
					dftval = dftval[-1]
	
				try:
					errval = 100*(float(potval)-float(dftval))/float(dftval)
				except ValueError:
					errval = ""
	
				self.data[key] = [potval, errval, dftval]
		
				f.seek(0,0)

			else:
				print "Couldn't find key " + key + " in file " + self.filename
				raise IOError
					

		f.close()

	def __eq__(self, other):

		if self.filename != other.filename:
			return False
		for key in self.data:
			if self.data[key] != other.data[key]:
				return False
		return True

	def __hash__(self):
		return hash(self.filename)	

	def get_filename(self):
		return self.filename
		
	def get_rms_error(self):
		rms = 0.0
		count = 0
		for key in self.data:
			if self.data[key][1] not in ["", np.inf]:
				count += 1
				rms += self.data[key][1]**2

		return np.sqrt(rms/count)
	
	def get_avg_error(self):
		avg = 0.0
		count = 0
		for key in self.data:
			if self.data[key][1] not in ["", np.inf]:
				count += 1
				avg += self.data[key][1]

		return avg/count

	def get_type_int(self):
		if self.get_filename().split(".")[0] == "MEAM":
			return 1
		elif self.get_filename().split(".")[0] == "EAM":
			return 0	

	def get_err_list(self):
		return [self.data[key][1] for key in self.data]
	
	def get_err_list_sub(self, keys):
		return [self.data[key][1] for key in keys]

	def print_pot(self):
		
		for key in self.data:
			print "%12s %8s %8s %8s" % (key, self.data[key][0], self.data[key][1], self.data[key][2])
