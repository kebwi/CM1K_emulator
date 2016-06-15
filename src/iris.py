from collections import defaultdict

"""
http:https://archive.ics.uci.edu/ml/datasets/Iris
"""


def read_iris(data_dir):
	data_grouped_by_cat = defaultdict(list)

	print "Reading IRIS dataset:"
	with open(data_dir + "iris.data.txt") as f:
		for line in f.readlines():
			parts = line.strip().split(',')
			assert (len(parts) == 5)  # SL (cm), SW (cm), PL (cm), PW (cm), class

			# IRIS features are presented in centimeters to one/tenth accuracy, e.g., 2.5, 7.3, 4.0, etc.
			# Convert them to millimeters and treat as type int.
			# Conveniently, all Iris data are under 256 millimeters and therefore can be represented in a single byte,
			# i.e., one CM1K pattern component.
			features = tuple([int(x.replace('.', '')) for x in parts[0:4]])
			cat = parts[4]
			# print "{}: {}".format(cat, features)
			data_grouped_by_cat[cat].append(features)

	return data_grouped_by_cat
