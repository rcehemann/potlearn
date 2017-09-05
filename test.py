#!/bin/python

import re

regex  = "(.*?EAM).(^[A-z]{1,2}).[^[A-z]{1,4}].[^[A-z]{1,4}].[^[\d]{1,2}].html"
regex  = "(^[a-zA-Z]{1,2}EAM)\.(.*?)\.(.*?)\.(.*?)\.(.*?)\.html"
filename = "MEAM.W.F.NC11.18.html"

print re.findall(regex, filename)

if re.match(regex, filename):
	print "fuck yeah"
