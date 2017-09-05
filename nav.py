#!/bin/python
import os
import re
def find_all(path):
	result = []
	regex  = "(^[a-zA-Z]{0,2}EAM)\.(.*?)\.(.*?)\.(.*?)\.(.*?)\.html"
	for root, dirs, files in os.walk(path):

			for fil in files:
				if re.match(regex, fil):
					result.append(root + "/" + fil)

	return result
