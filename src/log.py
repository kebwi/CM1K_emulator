"""
Keith Wiley
kwiley@keithwiley.com
http://keithwiley.com
"""

logging_enabled = False
trace_enabled = False
warning_enabled = True
error_enabled = True


def log(msg):
	if logging_enabled:
		print "LOG: ", msg


def trace(ftn):
	if trace_enabled:
		print "TRC: ", ftn


def warn(msg):
	if warning_enabled:
		print "WRN: ", msg


def error(msg):
	if error_enabled:
		print "ERR: ", msg
