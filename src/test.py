"""
Keith Wiley
kwiley@keithwiley.com
http://keithwiley.com
"""

import argparse
from collections import Counter
import math
import numpy as np
import random as rd
import pandas as pd
import log
import image_procs as img_p
import cm1k_emulator as cm1k
import neuron as nrn
import mnist
import att_db_of_faces
import iris
import mushroom


# Assign these to the database directories on disk
mnist_data_dir = None
faces_data_dir = None
iris_data_dir = None
mushroom_data_dir = None


def run_unit_test_01(verbose=0):
	"""
	Test a basic single CM1K chip network with a single context.
	"""
	if verbose >= 1:
		print "\n\n\n\n\ntest_minimal_01"

	network = cm1k.CM1KEmulator(network_size=4)
	assert(not network.euclidean_norm)
	network.write_maxif(2)

	train = [
		# Input, context, category
		("0000", 1, 1),
		("0000", 1, 1),  # Test repeating a training input
		("0011", 1, 2),
		("1110", 1, 3),
	]

	test = [
		# Input, context, categories (might be ambiguous)
		("0000", 1, (1,)),
		("0001", 1, (1, 2,)),
		("0010", 1, (1, 2,)),
		("0011", 1, (2,)),
		("0100", 1, (1,)),
		("0101", 1, ()),
		("0110", 1, (3,)),
		("0111", 1, (2,)),
		("1000", 1, (1,)),
		("1001", 1, ()),
		("1010", 1, (3,)),
		("1011", 1, (2,)),
		("1100", 1, (3,)),
		("1101", 1, ()),
		("1110", 1, (3,)),
		("1111", 1, (3,)),
	]

	# Train network
	for input_, cxt, cat in train:
		if verbose >= 1 and log.logging_enabled:
			print "================================================================================"
		input_comps = [int(x) for x in input_]
		network.learn(input_comps, cat, cxt)
	assert(network.read_ncount() == 3)

	# Test network
	if verbose >= 1:
		print "################################################################################"
	for test_input_, test_cxt, test_cats in test:
		if verbose >= 1:
			print "================================================================================"
			print "Test: {} {} {}".format(test_cxt, test_cats, test_input_)
		input_comps = [int(x) for x in test_input_]
		id_, unc, firing_neurons = network.broadcast(input_comps, test_cxt)
		best_neuron = firing_neurons[-1] if firing_neurons else None
		if verbose >= 1:
			print "id{} unc{} best_neuron: {}".format(id_, unc, best_neuron if best_neuron else None)
		cats = []
		dist = network.read_dist()
		while dist != 0xFFFF:
			cat = network.read_cat()
			cats.append(cat)
			nid = network.read_nid()
			if verbose >= 1:
				print "Next fire: nid{} dst{} cat{}".format(nid, dist, cat)
			dist = network.read_dist()
		if verbose >= 1:
			print "Testing: num categories: {} = {} ?".format(len(cats), len(test_cats))
		assert(len(cats) == len(test_cats))
		for i, cat in enumerate(cats):
			if verbose >= 1:
				print "Testing: category: {} = {} ?".format(cat, test_cats[i])
			assert(cat == test_cats[i])

	print "run_unit_test_01: PASS"


def run_unit_test_02(verbose=0):
	"""
	Test a network with unlimited size.
	It should add new neurons indefinitely as needed.
	"""
	if verbose >= 1:
		print "\n\n\n\n\ntest_minimal_02"

	network = cm1k.CM1KEmulator(network_size=0)
	assert(not network.euclidean_norm)
	network.write_maxif(2)

	train = [
		# Input, context, category
		("0000", 1, 1),
		("0000", 1, 1),  # Test repeating a training input
		("0011", 1, 2),
		("1110", 1, 3),
	]

	test = [
		# Input, context, categories (might be ambiguous)
		("0000", 1, (1,)),
		("0001", 1, (1, 2,)),
		("0010", 1, (1, 2,)),
		("0011", 1, (2,)),
		("0100", 1, (1,)),
		("0101", 1, ()),
		("0110", 1, (3,)),
		("0111", 1, (2,)),
		("1000", 1, (1,)),
		("1001", 1, ()),
		("1010", 1, (3,)),
		("1011", 1, (2,)),
		("1100", 1, (3,)),
		("1101", 1, ()),
		("1110", 1, (3,)),
		("1111", 1, (3,)),
	]

	# Train network
	for input_, cxt, cat in train:
		if verbose >= 1 and log.logging_enabled:
			print "================================================================================"
		input_comps = [int(x) for x in input_]
		network.learn(input_comps, cat, cxt)
	assert(network.read_ncount() == 3)

	# Test network
	if verbose >= 1:
		print "################################################################################"
	for test_input_, test_cxt, test_cats in test:
		if verbose >= 1:
			print "================================================================================"
			print "Test: {} {} {}".format(test_cxt, test_cats, test_input_)
		input_comps = [int(x) for x in test_input_]
		id_, unc, firing_neurons = network.broadcast(input_comps, test_cxt)
		best_neuron = firing_neurons[-1] if firing_neurons else None
		if verbose >= 1:
			print "id{} unc{} best_neuron: {}".format(id_, unc, best_neuron if best_neuron else None)
		cats = []
		dist = network.read_dist()
		while dist != 0xFFFF:
			cat = network.read_cat()
			cats.append(cat)
			nid = network.read_nid()
			if verbose >= 1:
				print "Next fire: nid{} dst{} cat{}".format(nid, dist, cat)
			dist = network.read_dist()
		if verbose >= 1:
			print "Testing: num categories: {} = {} ?".format(len(cats), len(test_cats))
		assert(len(cats) == len(test_cats))
		for i, cat in enumerate(cats):
			if verbose >= 1:
				print "Testing: category: {} = {} ?".format(cat, test_cats[i])
			assert(cat == test_cats[i])

	print "run_unit_test_02: PASS"


def run_unit_test_03(verbose=0):
	"""
	Test a network with multiple contexts.
	"""
	if verbose >= 1:
		print "\n\n\n\n\ntest_minimal_03"

	network = cm1k.CM1KEmulator(network_size=-1)
	assert(not network.euclidean_norm)
	network.write_maxif(2)

	train = [
		# Input, context, category
		("0000", 1, 1),
		("0000", 1, 1),  # Test repeating a training input
		("0011", 1, 2),
		("1110", 1, 3),
		("0000", 2, 11),
		("0000", 2, 11),  # Test repeating a training input
		("0011", 2, 12),
		("1110", 2, 13),
	]

	test = [
		# Input, context, categories (might be ambiguous)
		("0000", 1, (1,)),
		("0001", 1, (1, 2,)),
		("0010", 1, (1, 2,)),
		("0011", 1, (2,)),
		("0100", 1, (1,)),
		("0101", 1, ()),
		("0110", 1, (3,)),
		("0111", 1, (2,)),
		("1000", 1, (1,)),
		("1001", 1, ()),
		("1010", 1, (3,)),
		("1011", 1, (2,)),
		("1100", 1, (3,)),
		("1101", 1, ()),
		("1110", 1, (3,)),
		("1111", 1, (3,)),
		("0000", 2, (11,)),
		("0001", 2, (11, 12,)),
		("0010", 2, (11, 12,)),
		("0011", 2, (12,)),
		("0100", 2, (11,)),
		("0101", 2, ()),
		("0110", 2, (13,)),
		("0111", 2, (12,)),
		("1000", 2, (11,)),
		("1001", 2, ()),
		("1010", 2, (13,)),
		("1011", 2, (12,)),
		("1100", 2, (13,)),
		("1101", 2, ()),
		("1110", 2, (13,)),
		("1111", 2, (13,)),
	]

	# Train network
	for input_, cxt, cat in train:
		if verbose >= 1 and log.logging_enabled:
			print "================================================================================"
		input_comps = [int(x) for x in input_]
		network.learn(input_comps, cat, cxt)
	assert(network.read_ncount() == 6)

	# Test network
	if verbose >= 1:
		print "################################################################################"
	for test_input_, test_cxt, test_cats in test:
		if verbose >= 1:
			print "================================================================================"
			print "Test: {} {} {}".format(test_cxt, test_cats, test_input_)
		input_comps = [int(x) for x in test_input_]
		id_, unc, firing_neurons = network.broadcast(input_comps, test_cxt)
		best_neuron = firing_neurons[-1] if firing_neurons else None
		if verbose >= 1:
			print "id{} unc{} best_neuron: {}".format(id_, unc, best_neuron if best_neuron else None)
		cats = []
		dist = network.read_dist()
		while dist != 0xFFFF:
			cat = network.read_cat()
			cats.append(cat)
			nid = network.read_nid()
			if verbose >= 1:
				print "Next fire: nid{} dst{} cat{}".format(nid, dist, cat)
			dist = network.read_dist()
		if verbose >= 1:
			print "Testing: num categories: {} = {} ?".format(len(cats), len(test_cats))
		assert(len(cats) == len(test_cats))
		for i, cat in enumerate(cats):
			if verbose >= 1:
				print "Testing: category: {} = {} ?".format(cat, test_cats[i])
			assert(cat == test_cats[i])

	print "run_unit_test_03: PASS"


def run_unit_test_mnist_01(verbose=0):
	"""
	Run a few basic tests against a tiny subset of MNIST images
	"""
	if verbose >= 1:
		print "\n\n\n\n\nrun_unit_test_mnist_01"

	network = cm1k.CM1KEmulator(network_size=0)
	assert(not network.euclidean_norm)

	translation_sets = [
		((0, 0),),
	]

	# The first few labels from the MNIST train set file are, in order:
	# 5041921314
	train_set = mnist.read_mnist(
		mnist_data_dir, train_test="train", num_images_to_retrieve=10, first_image_to_retrieve=0,
		invert_images=False,
		crop_images_exclusion=None, center_images_exclusion=None,
		resampled_width=16, print_images=True if verbose >= 2 else False, write_transformed_image_files=False)

	# The first few labels from the MNIST test set file are, in order:
	# 7210414959
	test_set = mnist.read_mnist(
		mnist_data_dir, train_test="test", num_images_to_retrieve=10, first_image_to_retrieve=0,
		invert_images=False,
		crop_images_exclusion=None, center_images_exclusion=None,
		resampled_width=16, print_images=True if verbose >= 2 else False, write_transformed_image_files=False)

	if verbose >= 1:
		print "TRAIN LABELS: {}".format(' '.join([str(x[1]) for x in train_set]))
		print "TEST LABELS:  {}".format(' '.join([str(x[1]) for x in test_set]))

	# Train the network
	cat_counts_sorted_by_val = train_mnist_or_att_faces(0, train_set, network, 0)

	# Classify the test dataset with the trained network
	legend, row = test_mnist_or_att_faces(
		cat_counts_sorted_by_val, 1, translation_sets,
		False, None, None, 16, False, None,
		5000, len(test_set), test_set, network, 0
	)

	correct_results = ['N', 'None', 'None', '16', 'N', 'None', '5000', '10', '8', '0.125', '0.125', '0.125', '0.125',
					   '0.25', '0.125', '0.125', '1', '3', '5', '2', '0', '7', '0', '0', '0', '0', '0', '0', '5', '3',
					   '0', '2', '0', '2', '2', '0', '0', '0', '3', '0', '2', '0', '2', '0', '0', '0']

	if verbose >= 1:
		print "Testing: results = correct results ?"
		print "Correct results: ", correct_results
		print "Actual results:  ", row

	assert row == correct_results

	print "run_unit_test_mnist_01: PASS"


def train_mnist_or_att_faces(train_set_offset, train_set, network, cxt, verbose=0):
	"""
	This function should be adaptable to image-classification tasks. The two datasets I used during development are already
	8-bit grayscale. All I had to do was downsample the images to 16x16 pixels so as to fit in the CM1K's 256 byte max.
	pattern length. Alternatively, another CM1K approach is to classify larger images as multiple 16x16 tiles with a
	separate CM1K context assigned to each tile. The multi-tile classifications must then be aggregated (voted) somehow.
	"""
	if verbose >= 1:
		print "################################################################################"
		print "TRAIN with {} images".format(len(train_set))
		print "{:>5}\t{:>5}\t{:>5}\t{:>5}\t{:>5}\t{:>10}\t{:>10}\t{:>10}".format(
			"Idx", "Comm", "Degen", "AIFmn", "AIFmx", "AIFmean", "AIFstdv", "AIFmedian")
	for i, (img, lbl) in enumerate(train_set):
		if verbose >= 1:
			# As training instances are incrementally presented, occasionally dump some ongoing metrics
			if i > 0 and (i < 10 or (i < 100 and i % 10 == 0) or (i < 1000 and i % 100 == 0) or i % 1000 == 0):
				num_committed_neurons = network.read_ncount()
				num_degenerate = 0
				aifs = []
				for neuron in network.neurons:
					if neuron.state == nrn.NeuronState.com:
						if neuron.degenerate:
							num_degenerate += 1
						aifs.append(neuron.aif)
				assert(len(aifs) == num_committed_neurons)
				aif_min = np.min(aifs)
				aif_max = np.max(aifs)
				aif_mean = np.mean(aifs)
				aif_stddev = np.std(aifs)
				aif_median = np.median(aifs)

				print "{:>5}\t{:>5}\t{:>5}\t{:>5}\t{:>5}\t{:>10.1f}\t{:>10.1f}\t{:>10.1f}".format(
					train_set_offset + i, num_committed_neurons, num_degenerate, aif_min, aif_max, aif_mean, aif_stddev, aif_median)

		# print img
		# print lbl
		input_comps = [ord(x) for x in img]  # Convert input components to ordinal byte values
		# print_ascii_image(input_comps, 16)
		network.learn(input_comps, lbl, cxt)

	# Dump a final set of metrics after all training instances have been presented
	num_committed_neurons = network.read_ncount()
	num_degenerate = 0
	aifs = []
	cat_counts = Counter()
	for neuron in network.neurons:
		if neuron.state == nrn.NeuronState.com:
			if neuron.degenerate:
				num_degenerate += 1
			aifs.append(neuron.aif)
			cat_counts[neuron.cat] += 1
	assert(len(aifs) == num_committed_neurons)
	aif_min = np.min(aifs)
	aif_max = np.max(aifs)
	aif_mean = np.mean(aifs)
	aif_stddev = np.std(aifs)
	aif_median = np.median(aifs)
	aifs_hist = np.histogram(aifs)

	cat_counts_sorted = cat_counts.most_common()
	cat_counts_sorted_by_val = sorted(cat_counts_sorted, key=lambda x: x[0])

	if verbose >= 1:
		print "{:>5}\t{:>5}\t{:>5}\t{:>5}\t{:>5}\t{:>10.1f}\t{:>10.1f}\t{:>10.1f}".format(
			train_set_offset + len(train_set), num_committed_neurons, num_degenerate, aif_min, aif_max, aif_mean, aif_stddev, aif_median)

	if verbose >= 1:
		# If the number of committed neurons is small, dump them all
		if len(aifs) < 1000:
			print "{} AIFS:".format(len(aifs))
			print ' '.join([str(x) for x in sorted(aifs)])
		else:
			print "Didn't print full AIF list because there are {}".format(len(aifs))

		# Dump the AIF histogram
		print "AIF histogram:"
		print list(aifs_hist[0])
		print list(aifs_hist[1])

		# Dump the distribution of the number of committed neurons per category, both in raw counts and proportionally
		print "Committed neuron distribution:"
		for cat_count in cat_counts_sorted_by_val:
			print "{:>2}\t{:>5}".format(cat_count[0], cat_count[1])
		for cat_count in cat_counts_sorted_by_val:
			print "{:>2}\t{:>5}".format(cat_count[0], float(cat_count[1]) / num_committed_neurons)

	return cat_counts_sorted_by_val


def test_mnist_or_att_faces_out(
		invert_images, crop_images_exclusion, center_images_exclusion, resampled_width, randomize_train_set, rseed,
		num_training_images, num_testing_images,
		network, cat_counts_sorted_by_val, aif_scale, translation_sets,
		num_id0_unc0, num_id1_unc0, num_id0_unc1, num_id1_unc1,
		matches_per_trans_set,
		pos_mode_vote_trans0_sum, neg_mode_vote_trans0_sum, pos_mode_vote_sum, neg_mode_vote_sum,
		num_correct_id_id, num_correct_id_unc, num_correct_unc_id, num_correct_unc_unc,
		num_wrong_id_id, num_wrong_id_unc, num_wrong_unc_id, num_wrong_unc_unc,
		num_correct_id_id_per_trans_set_idx, num_correct_id_unc_per_trans_set_idx,
		num_correct_unc_id_per_trans_set_idx, num_correct_unc_unc_per_trans_set_idx,
		num_wrong_id_id_per_trans_set_idx, num_wrong_id_unc_per_trans_set_idx,
		num_wrong_unc_id_per_trans_set_idx, num_wrong_unc_unc_per_trans_set_idx,
		verbose=0):
	"""
	Dump the results of classification of a test set using a trained network
	"""
	if verbose >= 1:
		# Dump all the results to the screen, with labels, for maximum legibility
		print "################################################################################"

		print "Results:"
		print "Invert:                {}".format("Y" if invert_images else "N")
		print "Crop:                  {}".format(crop_images_exclusion)
		print "Center:                {}".format(center_images_exclusion)
		print "Resampled width:       {}".format(resampled_width)
		print "Randomize:             {}".format("Y" if randomize_train_set else "N")
		print "Random seed:           {}".format(rseed)
		print "Train set size:        {}".format(num_training_images)
		print "Test set size:         {}".format(num_testing_images)
		print "Num committed:         {}".format(network.read_ncount())
		print "Committed distribution:"
		for k, v in cat_counts_sorted_by_val:
			print "  {}: {}".format(k, v)
		for k, v in cat_counts_sorted_by_val:
			print "  {}: {}".format(k, float(v) / network.read_ncount())
		print "AIF scale:             {}".format(aif_scale)
		print "num_id0_unc0:          {}".format(num_id0_unc0)
		print "num_id1_unc0:          {}".format(num_id1_unc0)
		print "num_id0_unc1:          {}".format(num_id0_unc1)
		print "num_id1_unc1:          {}".format(num_id1_unc1)
		print "matches_per_trans_set: {}".format(matches_per_trans_set)
		for i in xrange(len(translation_sets)):
			print "  {}: {}".format(i, matches_per_trans_set[i][1])
		print "pos_mode_vote_trans0_sum: {}".format(pos_mode_vote_trans0_sum)
		print "neg_mode_vote_trans0_sum: {}".format(neg_mode_vote_trans0_sum)
		print "tot_mode_vote_trans0_sum: {}".format(pos_mode_vote_trans0_sum + neg_mode_vote_trans0_sum)
		print "pos_mode_vote_sum:        {}".format(pos_mode_vote_sum)
		print "neg_mode_vote_sum:        {}".format(neg_mode_vote_sum)
		print "tot_mode_vote_sum:        {}".format(pos_mode_vote_sum + neg_mode_vote_sum)
		print "num correct total:     {}".format(num_correct_id_id + num_correct_id_unc + num_correct_unc_id + num_correct_unc_unc)
		print "num_correct_id_id:     {}".format(num_correct_id_id)
		print "num_correct_id_unc:    {}".format(num_correct_id_unc)
		print "num_correct_unc_id:    {}".format(num_correct_unc_id)
		print "num_correct_unc_unc:   {}".format(num_correct_unc_unc)
		print "num_correct_id_id_per_trans_set_idx:   {}".format(num_correct_id_id_per_trans_set_idx)
		print "num_correct_id_unc_per_trans_set_idx:  {}".format(num_correct_id_unc_per_trans_set_idx)
		print "num_correct_unc_id_per_trans_set_idx:  {}".format(num_correct_unc_id_per_trans_set_idx)
		print "num_correct_unc_unc_per_trans_set_idx: {}".format(num_correct_unc_unc_per_trans_set_idx)
		for i in xrange(len(translation_sets)):
			print "  {} id_id:   {}".format(i, num_correct_id_id_per_trans_set_idx[i][1])
			print "  {} id_unc:  {}".format(i, num_correct_id_unc_per_trans_set_idx[i][1])
			print "  {} unc_id:  {}".format(i, num_correct_unc_id_per_trans_set_idx[i][1])
			print "  {} unc_unc: {}".format(i, num_correct_unc_unc_per_trans_set_idx[i][1])
		print "num wrong total:       {}".format(num_wrong_id_id + num_wrong_id_unc + num_wrong_unc_id + num_wrong_unc_unc)
		print "num_wrong_id_id:       {}".format(num_wrong_id_id)
		print "num_wrong_id_unc:      {}".format(num_wrong_id_unc)
		print "num_wrong_unc_id:      {}".format(num_wrong_unc_id)
		print "num_wrong_unc_unc:     {}".format(num_wrong_unc_unc)
		print "num_wrong_id_id_per_trans_set_idx:   {}".format(num_wrong_id_id_per_trans_set_idx)
		print "num_wrong_id_unc_per_trans_set_idx:  {}".format(num_wrong_id_unc_per_trans_set_idx)
		print "num_wrong_unc_id_per_trans_set_idx:  {}".format(num_wrong_unc_id_per_trans_set_idx)
		print "num_wrong_unc_unc_per_trans_set_idx: {}".format(num_wrong_unc_unc_per_trans_set_idx)
		for i in xrange(len(translation_sets)):
			print "  {} id_id:   {}".format(i, num_wrong_id_id_per_trans_set_idx[i][1])
			print "  {} id_unc:  {}".format(i, num_wrong_id_unc_per_trans_set_idx[i][1])
			print "  {} unc_unc: {}".format(i, num_wrong_unc_id_per_trans_set_idx[i][1])
			print "  {} unc_unc: {}".format(i, num_wrong_unc_unc_per_trans_set_idx[i][1])

	# Create a legend that can easily be copy/pasted into a spreadsheet
	legend = [
		'INVERT IMAGES',
		'CROP IMAGES',
		'CENTER IMAGES',
		'RESAMPLED IMAGES',
		'RANDOMIZE',
		'RANDOM SEED',
		'TRAIN SET SIZE',
		'TEST SET SIZE',
		'COMMITTED NEURONS',
	]
	for k, v in cat_counts_sorted_by_val:
		legend.append('COMMITTED NEURONS {}'.format(k))
	legend.extend([
		'AIF SCALE',
		'NUM ID0 UNC0',
		'NUM ID1 UNC0',
		'NUM ID0 UNC1',
		'NUM ID1 UNC1',
	])
	for i in xrange(len(translation_sets)):
		legend.append('TRANS SET {}'.format(i))
	legend.extend([
		'POS MODE VOTE TS0 SUM',
		'NEG MODE VOTE TS0 SUM',
		'TOT MODE VOTE TS0 SUM',
		'POS MODE VOTE SUM',
		'NEG MODE VOTE SUM',
		'TOT MODE VOTE SUM',
		'NUM CORRECT TOTAL',
		'NUM CORRECT ID ID',
		'NUM CORRECT ID UNC',
		'NUM CORRECT UNC ID',
		'NUM CORRECT UNC UNC',
		'NUM WRONG TOTAL',
		'NUM WRONG ID ID',
		'NUM WRONG ID UNC',
		'NUM WRONG UNC ID',
		'NUM WRONG UNC UNC',
	])
	for i in xrange(len(translation_sets)):
		legend.append('NUM CORRECT TRANS SET {} ID ID'.format(i))
		legend.append('NUM CORRECT TRANS SET {} ID UNC'.format(i))
		legend.append('NUM CORRECT TRANS SET {} UNC ID'.format(i))
		legend.append('NUM CORRECT TRANS SET {} UNC UNC'.format(i))
	for i in xrange(len(translation_sets)):
		legend.append('NUM WRONG TRANS SET {} ID ID'.format(i))
		legend.append('NUM WRONG TRANS SET {} ID UNC'.format(i))
		legend.append('NUM WRONG TRANS SET {} UNC ID'.format(i))
		legend.append('NUM WRONG TRANS SET {} UNC UNC'.format(i))

	# Create an output array that matches the legend (above) that can easily be copy/pasted into a spreadsheet
	output = []
	output.append("{}".format("Y" if invert_images else "N"))
	output.append("{}".format(crop_images_exclusion))
	output.append("{}".format(center_images_exclusion))
	output.append("{}".format(resampled_width))
	output.append("{}".format("Y" if randomize_train_set else "N"))
	output.append("{}".format(rseed))
	output.append("{}".format(num_training_images))
	output.append("{}".format(num_testing_images))
	output.append("{}".format(network.read_ncount()))
	for k, v in cat_counts_sorted_by_val:
		# It's probably more useful to compare proportional distribution regardless of training set size,
		# so we probably want to use the second line here, not the first.
		# output.append("{}".format(v))
		output.append("{}".format(float(v) / network.read_ncount()))
	output.append("{}".format(aif_scale))
	output.append("{}".format(num_id0_unc0))
	output.append("{}".format(num_id1_unc0))
	output.append("{}".format(num_id0_unc1))
	output.append("{}".format(num_id1_unc1))
	for i in xrange(len(translation_sets)):
		output.append("{}".format(matches_per_trans_set[i][1]))
	output.append("{}".format(pos_mode_vote_trans0_sum))
	output.append("{}".format(neg_mode_vote_trans0_sum))
	output.append("{}".format(pos_mode_vote_trans0_sum + neg_mode_vote_trans0_sum))
	output.append("{}".format(pos_mode_vote_sum))
	output.append("{}".format(neg_mode_vote_sum))
	output.append("{}".format(pos_mode_vote_sum + neg_mode_vote_sum))
	output.append("{}".format(num_correct_id_id + num_correct_id_unc + num_correct_unc_id + num_correct_unc_unc))
	output.append("{}".format(num_correct_id_id))
	output.append("{}".format(num_correct_id_unc))
	output.append("{}".format(num_correct_unc_id))
	output.append("{}".format(num_correct_unc_unc))
	output.append("{}".format(num_wrong_id_id + num_wrong_id_unc + num_wrong_unc_id + num_wrong_unc_unc))
	output.append("{}".format(num_wrong_id_id))
	output.append("{}".format(num_wrong_id_unc))
	output.append("{}".format(num_wrong_unc_id))
	output.append("{}".format(num_wrong_unc_unc))
	for i in xrange(len(translation_sets)):
		output.append("{}".format(num_correct_id_id_per_trans_set_idx[i][1]))
		output.append("{}".format(num_correct_id_unc_per_trans_set_idx[i][1]))
		output.append("{}".format(num_correct_unc_id_per_trans_set_idx[i][1]))
		output.append("{}".format(num_correct_unc_unc_per_trans_set_idx[i][1]))
	for i in xrange(len(translation_sets)):
		output.append("{}".format(num_wrong_id_id_per_trans_set_idx[i][1]))
		output.append("{}".format(num_wrong_id_unc_per_trans_set_idx[i][1]))
		output.append("{}".format(num_wrong_unc_id_per_trans_set_idx[i][1]))
		output.append("{}".format(num_wrong_unc_unc_per_trans_set_idx[i][1]))

	assert len(legend) == len(output)

	if verbose >= 1:
		print "Legend ({}):".format(len(legend))
		print '\n'.join(legend)
		print "Results ({}):".format(len(output))
		print '\n'.join(output)

	return legend, output


def test_mnist_or_att_faces(
		cat_counts_sorted_by_val, aif_scale, translation_sets,
		invert_images, crop_images_exclusion, center_images_exclusion, resampled_width, randomize_train_set, rseed,
		num_training_images, num_testing_images, test_set, network, cxt, verbose=0):
	"""
	Classify a test dataset with a trained network
	"""
	if verbose >= 1:
		print "################################################################################"
		print "TEST with {} images".format(num_testing_images)
	assert(len(test_set) == num_testing_images)
	bg_color = 255 if invert_images else 0

	# There are four ID/UNC pair combinations:
	# ID ID:   Unique identification at all translations and across best neurons of all translations
	# ID UNC:  Unique identification at all translations, but uncertainty across best neurons of some translations
	# UNC ID:  Uncertain identification at some translations, but unique across best neurons of all translations
	# UNC UNC: Uncertain identification at some translations and across best neurons of some translations

	num_id0_unc0 = 0
	num_id1_unc0 = 0
	num_id0_unc1 = 0
	num_id1_unc1 = 0
	num_correct_id_id = 0
	num_correct_id_unc = 0
	num_correct_unc_id = 0
	num_correct_unc_unc = 0
	num_wrong_id_id = 0
	num_wrong_id_unc = 0
	num_wrong_unc_id = 0
	num_wrong_unc_unc = 0

	matches_per_trans_set = Counter()
	num_correct_id_id_per_trans_set_idx = Counter()
	num_correct_id_unc_per_trans_set_idx = Counter()
	num_correct_unc_id_per_trans_set_idx = Counter()
	num_correct_unc_unc_per_trans_set_idx = Counter()
	num_wrong_id_id_per_trans_set_idx = Counter()
	num_wrong_id_unc_per_trans_set_idx = Counter()
	num_wrong_unc_id_per_trans_set_idx = Counter()
	num_wrong_unc_unc_per_trans_set_idx = Counter()
	for i in xrange(len(translation_sets)):
		matches_per_trans_set[i] = 0
		num_correct_id_id_per_trans_set_idx[i] = 0
		num_correct_id_unc_per_trans_set_idx[i] = 0
		num_correct_unc_id_per_trans_set_idx[i] = 0
		num_correct_unc_unc_per_trans_set_idx[i] = 0
		num_wrong_id_id_per_trans_set_idx[i] = 0
		num_wrong_id_unc_per_trans_set_idx[i] = 0
		num_wrong_unc_id_per_trans_set_idx[i] = 0
		num_wrong_unc_unc_per_trans_set_idx[i] = 0

	# Keep track of how mode-voting would alter the default winner-takes-all classification results.
	# Mode-voting can help (correcting a classification which winner-takes-all classified wrong),
	# or it can hurt (classifying wrongly where winner-takes-all would have been correct).
	pos_mode_vote_sum = 0  # Across all translation sets, how many mode-votes correct a wrong winner-takes-all vote
	neg_mode_vote_sum = 0
	pos_mode_vote_trans0_sum = 0  # Keep track of translation set 0, so we can evaluate the usefulness of translations.
	neg_mode_vote_trans0_sum = 0

	# Iterate over the test set
	for img_idx, (img, lbl) in enumerate(test_set):
		# print "================================================================================"
		if verbose >= 1:
			if img_idx % 100 == 0:
				print "Idx: {}".format(img_idx)
			# print img
			# print lbl
		input_comps = [ord(x) for x in img]  # Convert input components to ordinal byte values
		# print_ascii_image(input_comps, 16)

		# Iterate over the translation sets. Once a classification is made, no deeper translation sets will be considered.
		for trans_set_idx, translation_set in enumerate(translation_sets):
			matches = []  # A list of lists of (cat, dist), maintained in order of descending dist of the best neuron in each list
			uncertain = False  # Did any translation within the set produce an uncertain result?
			# Iterate over the translations within a single translation set
			for translation in translation_set:
				# Shift the image according to the translation
				shifted_img = img_p.shift_image(
					input_comps, resampled_width, resampled_width, translation[0], translation[1], bg_color)
				# print "id{} unc{} best_neuron: {}".format(id_, unc, best_neuron if best_neuron else None)

				# Broadcast the image to the network and retrieve the classification results
				id_, unc, firing_neurons = network.broadcast(shifted_img, cxt, False, aif_scale)
				firing_neuron_cat_dists = [(x.cat, x.dist) for x in firing_neurons]
				best_neuron = firing_neurons[-1] if firing_neurons else None

				assert(id_ == 0 or unc == 0)  # id and unc can't both be true
				if id_ != 0 or unc != 0:  # If id or unc is true, there must be a best neuron
					assert best_neuron
				if best_neuron:  # If there is a best neuron, id or unc must be true
					assert(id_ != 0 or unc != 0)

				# If any neurons fired, gather data on the strongest firing neuron
				if best_neuron:  # id_ != 0 or unc != 0:
					if id_ == 0 and unc == 1:
						uncertain = True

					dist = network.read_dist()
					assert(dist == best_neuron.dist)
					insert_pos = len(matches)
					for match_idx, match in enumerate(matches):
						if match[1][-1][1] < dist:
							insert_pos = match_idx
							break
					assert firing_neuron_cat_dists
					matches.insert(insert_pos, (translation, firing_neuron_cat_dists))

			# Get the set of all matched categories across all neurons that fired for this translation set
			matched_all_cats = set()
			num_firing_neurons = 0
			for match in matches:
				num_firing_neurons += len(match[1])
				for cat_dist in match[1]:
					matched_all_cats.add(cat_dist[0])

			# Get the set of strongest firing neurons across all translations (one per translations)
			matched_best_cats = set([x[1][-1][0] for x in matches])

			# Get the mode classification (which category got the most hits)
			cat_mode = None
			if num_firing_neurons > 0:
				cat_occurrences = Counter()
				for match in matches:
					for cat_dist in match[1]:
						cat_occurrences[cat_dist[0]] += 1
				cat_occurrences_sorted = cat_occurrences.most_common()
				if len(cat_occurrences_sorted) >= 2 and cat_occurrences_sorted[0][1] > cat_occurrences_sorted[1][1]:
					cat_mode = cat_occurrences_sorted[0][0]

			# if len(matched_best_cats) > 0 and num_firing_neurons > 2:
			# 	print ">2 {:0>5}       lbl:{}    #bcts:{}    #acts:{}    bst:{}    ctm:{:>4}    tsi:{}".format(
			# 		img_idx, lbl, len(matched_best_cats), len(matched_all_cats), matches[-1][1][-1][0], cat_mode, trans_set_idx)
			# 	for match in reversed(matches):
			# 		fn_out = "    ({:>2},{:>2}) =>".format(match[0][0], match[0][1])
			# 		for cat_dist in reversed(match[1]):
			# 			fn_out += "    {}".format(cat_dist)
			# 		print fn_out

			# There are three cases in this outer level condition:
			# (1) No neurons fired across all translations (of this translation set)
			# (2) The best results across all translations of this set agree
			# (3) The best results across all translations disagree
			if len(matched_best_cats) == 0 and trans_set_idx == len(translation_sets) - 1:
				# Unclassified: no neurons fired across all translations (of this translation set)
				num_id0_unc0 += 1
				break  # Not really needed since we know we're at the end of the loop, but let's maintain the pattern
			elif len(matched_best_cats) == 1:  # The best results across all translations of this set agree
				status = 0  # The id/unc correct/wrong status
				mode_effect = 0  # 1: cat_mode would correct a wrong classification, -1: it would wrong a correct one

				if not uncertain:  # At each translation, all firing neurons agree on the category
					num_id1_unc0 += 1
					if matches[-1][1][-1][0] == lbl:   # Correct
						num_correct_id_id += 1
						num_correct_id_id_per_trans_set_idx[trans_set_idx] += 1
						status = 1
						if cat_mode is not None and cat_mode != lbl:
							mode_effect = -1
					else:  # Wrong
						num_wrong_id_id += 1
						num_wrong_id_id_per_trans_set_idx[trans_set_idx] += 1
						status = 2
						if cat_mode is not None and cat_mode == lbl:
							mode_effect = 1
				else:  # At some translations, the firing neurons disagree on the category
					num_id0_unc1 += 1
					if matches[-1][1][-1][0] == lbl:   # Correct
						num_correct_unc_id += 1
						num_correct_unc_id_per_trans_set_idx[trans_set_idx] += 1
						status = 3
						if cat_mode is not None and cat_mode != lbl:
							mode_effect = -1
					else:  # Wrong
						num_wrong_unc_id += 1
						num_wrong_unc_id_per_trans_set_idx[trans_set_idx] += 1
						status = 4
						if cat_mode is not None and cat_mode == lbl:
							mode_effect = 1
				matches_per_trans_set[trans_set_idx] += 1

				# Keep track of the effect that mode-voting would have relative to winner-takes-all
				if mode_effect != 0:
					if mode_effect > 0:
						pos_mode_vote_sum += mode_effect
						if len(matches) == 1 and matches[0] == (0, 0):
							pos_mode_vote_trans0_sum += mode_effect
					else:
						neg_mode_vote_sum += mode_effect
						if len(matches) == 1 and matches[0] == (0, 0):
							neg_mode_vote_trans0_sum += mode_effect
					print "{:0>5}          lbl:{}    #bcts:{}    #acts:{}    ctm:{:>4}    tsi:{}    sts:{}    mef:{}".format(
						img_idx, lbl, len(matched_best_cats), len(matched_all_cats), cat_mode, trans_set_idx, status, mode_effect)
					for match in reversed(matches):
						fn_out = "    ({:>2},{:>2}) =>".format(match[0][0], match[0][1])
						for cat_dist in reversed(match[1]):
							fn_out += "    {}".format(cat_dist)
						print fn_out
				break
			elif len(matched_best_cats) > 1:  # The best results across all translations disagree
				status = 10  # The id/unc correct/wrong status
				mode_effect = 0  # 10: cat_mode would correct a wrong classification, -10: it would wrong a correct one

				num_id0_unc1 += 1
				if not uncertain:  # At each translation, all firing neurons agree on the category
					if matches[-1][1][-1][0] == lbl:   # Correct
						num_correct_id_unc += 1
						num_correct_id_unc_per_trans_set_idx[trans_set_idx] += 1
						status = 11
						if cat_mode is not None and cat_mode != lbl:
							mode_effect = -10
					else:  # Wrong
						num_wrong_id_unc += 1
						num_wrong_id_unc_per_trans_set_idx[trans_set_idx] += 1
						status = 12
						if cat_mode is not None and cat_mode == lbl:
							mode_effect = 10
				else:  # At some translations, the firing neurons disagree on the category
					if matches[-1][1][-1][0] == lbl:   # Correct
						num_correct_unc_unc += 1
						num_correct_unc_unc_per_trans_set_idx[trans_set_idx] += 1
						status = 13
						if cat_mode is not None and cat_mode != lbl:
							mode_effect = -10
					else:  # Wrong
						num_wrong_unc_unc += 1
						num_wrong_unc_unc_per_trans_set_idx[trans_set_idx] += 1
						status = 14
						if cat_mode is not None and cat_mode == lbl:
							mode_effect = 10
				matches_per_trans_set[trans_set_idx] += 1

				# Keep track of the effect that mode-voting would have relative to winner-takes-all
				if mode_effect != 0:
					if mode_effect > 0:
						pos_mode_vote_sum += mode_effect / 10
						if len(matches) == 1 and matches[0] == (0, 0):
							pos_mode_vote_trans0_sum += mode_effect / 10
					else:
						neg_mode_vote_sum += mode_effect / 10
						if len(matches) == 1 and matches[0] == (0, 0):
							neg_mode_vote_trans0_sum += mode_effect / 10
					print "{:0>5}          lbl:{}    #bcts:{}    #acts:{}    tsi:{}   sts:{}    mef:{}".format(
						img_idx, lbl, len(matched_best_cats), len(matched_all_cats), trans_set_idx, status, mode_effect)
					for match in reversed(matches):
						fn_out = "    ({:>2},{:>2}) =>".format(match[0][0], match[0][1])
						for cat_dist in reversed(match[1]):
							fn_out += "    {}".format(cat_dist)
						print fn_out
				break

	# print "pos_mv_t0_sum: {}".format(pos_mode_vote_trans0_sum)
	# print "neg_mv_t0_sum: {}".format(neg_mode_vote_trans0_sum)
	# print "tot_mv_t0_sum: {}".format(pos_mode_vote_trans0_sum + neg_mode_vote_trans0_sum)
	# print "pos_mv_sum:    {}".format(pos_mode_vote_sum)
	# print "neg_mv_sum:    {}".format(neg_mode_vote_sum)
	# print "tot_mv_sum:    {}".format(pos_mode_vote_sum + neg_mode_vote_sum)

	# Gather up the more nested metrics
	matches_per_trans_set = sorted(matches_per_trans_set.items())
	num_correct_id_id_per_trans_set_idx = sorted(num_correct_id_id_per_trans_set_idx.items())
	num_correct_id_unc_per_trans_set_idx = sorted(num_correct_id_unc_per_trans_set_idx.items())
	num_correct_unc_id_per_trans_set_idx = sorted(num_correct_unc_id_per_trans_set_idx.items())
	num_correct_unc_unc_per_trans_set_idx = sorted(num_correct_unc_unc_per_trans_set_idx.items())
	num_wrong_id_id_per_trans_set_idx = sorted(num_wrong_id_id_per_trans_set_idx.items())
	num_wrong_id_unc_per_trans_set_idx = sorted(num_wrong_id_unc_per_trans_set_idx.items())
	num_wrong_unc_id_per_trans_set_idx = sorted(num_wrong_unc_id_per_trans_set_idx.items())
	num_wrong_unc_unc_per_trans_set_idx = sorted(num_wrong_unc_unc_per_trans_set_idx.items())

	# Dump the results
	legend, output = test_mnist_or_att_faces_out(
		invert_images, crop_images_exclusion, center_images_exclusion, resampled_width, randomize_train_set, rseed,
		num_training_images, num_testing_images, network, cat_counts_sorted_by_val, aif_scale, translation_sets,
		num_id0_unc0, num_id1_unc0, num_id0_unc1, num_id1_unc1,
		matches_per_trans_set,
		pos_mode_vote_trans0_sum, neg_mode_vote_trans0_sum, pos_mode_vote_sum, neg_mode_vote_sum,
		num_correct_id_id, num_correct_id_unc, num_correct_unc_id, num_correct_unc_unc,
		num_wrong_id_id, num_wrong_id_unc, num_wrong_unc_id, num_wrong_unc_unc,
		num_correct_id_id_per_trans_set_idx, num_correct_id_unc_per_trans_set_idx,
		num_correct_unc_id_per_trans_set_idx, num_correct_unc_unc_per_trans_set_idx,
		num_wrong_id_id_per_trans_set_idx, num_wrong_id_unc_per_trans_set_idx,
		num_wrong_unc_id_per_trans_set_idx, num_wrong_unc_unc_per_trans_set_idx,
		verbose
	)

	return legend, output


def run_oneconfig_mnist(
		invert_images, crop_images_exclusion, center_images_exclusion, resampled_width, euclidean_norm,
		use_translation_sets, num_training_images, num_testing_images, network_size):
	"""
	This test relies on the MNIST database. You must supply paths to the files before this test can run.
	The paths are currently hard-coded into the correspoding source file.
	"""
	network = cm1k.CM1KEmulator(network_size=network_size)

	# The CM1K doesn't support a Euclidean distance norm, but the emulator offers it for comparison purposes.
	# If indicated, set the network to use the Euclidean distance norm.
	if euclidean_norm:
		network.euclidean_norm = True
		# 0x400 would indicate the maximum conceivable Euclidean distance for a 256-byte pattern.
		# 0x100 would indicate one fourth the max. AIF, which corresponds to the fact that the CM1K's default max. AIF
		# for its included L1 dist. norm. is one fourth of the conceivable distance.
		euclidean_max_aif = euclidean_norm
		network.write_maxif(euclidean_max_aif)
	print "Eucliean norm: {}".format(network.euclidean_norm)
	print "Max AIF:       {}".format(network.read_maxif())

	# I briefly experimented with scaling all AIFs up a little bit after training was complete, in an effort to decrease
	# unclassifications during the test stage. However, this idea did't work out well. While it had the desired effect,
	# it always increased wrong classifications more than correct classifications. I recommend leaving this set to 1.
	aif_scale = 1

	# If indicated to use translation sets, initialize them.
	# If indicated not to, create only the first translation set, which is "no translation".
	# The translation sets describe sets of "chess-board-squares" of incrementally increasing distance from the center.
	if use_translation_sets:
		translation_sets = [
			((0, 0),),  # No translation
			((0, -1), (0, 1), (-1, 0), (1, 0)),  # Cardinal neighbors
			((-1, -1), (-1, 1), (1, -1), (1, 1)),  # Diagonal neighbors
			((0, -2), (0, 2), (-2, 0), (2, 0)),  # Cardinal neighbors two steps out
			((1, -2), (1, 2), (-2, 1), (2, 1), (-1, -2), (-1, 2), (-2, -1), (2, -1)),  # Knight's-move neighbors
			# There is no point in considering further translations. Experiments demonstrated that the benefits of
			# translating an image to find a match against the network's prototypes drop off very quickly relative to
			# the translation distance.
		]
	else:
		translation_sets = [
			((0, 0),),
		]
	print "Num translation sets: {}".format(len(translation_sets))

	# Since the MNIST database is stored with inverted brightness, I experimented with inverting it as a pre-processing
	# stage. One would expect this to have no mathematical effect, and the experiments confirmed this.  I recommend not
	# bothering to enable the inversion parameter.
	print "Invert images:    {}".format(invert_images)
	# Crop exclusion indicates the pixel value threshold used when determining a cropping-box.
	print "Crop exclusion:   {}".format(crop_images_exclusion)
	# Center exclusion indicates the pixel value threshold used when determining a centering-box.
	print "Center exclusion: {}".format(center_images_exclusion)

	# Read the test MNSIT dataset (or some subset of it)
	test_set = mnist.read_mnist(
		mnist_data_dir, train_test="test", num_images_to_retrieve=num_testing_images, first_image_to_retrieve=0,
		invert_images=invert_images,
		crop_images_exclusion=crop_images_exclusion, center_images_exclusion=center_images_exclusion,
		resampled_width=resampled_width, print_images=False, write_transformed_image_files=False)

	cxt = 1

	ideal_batch_size = 5000
	if True:  # args.command_line_flag:
		training_batch_size = min(ideal_batch_size, num_training_images)
		num_training_batches = int(math.ceil(float(num_training_images) / training_batch_size))
	else:
		training_batch_size = num_training_images
		num_training_batches = 1

	print "Training batch size:  {}".format(training_batch_size)
	print "Num training batches: {}".format(num_training_batches)

	legend = None
	results_table = []
	for i in xrange(0, num_training_batches):
		print "\n\n\n"
		print "Training batch: {}".format(i)

		# Read the train dataset (or some subset of it)
		train_set = mnist.read_mnist(
			mnist_data_dir, train_test="train", num_images_to_retrieve=training_batch_size,
			first_image_to_retrieve=i * training_batch_size,
			invert_images=invert_images,
			crop_images_exclusion=crop_images_exclusion, center_images_exclusion=center_images_exclusion,
			resampled_width=resampled_width,
			print_images=False, write_transformed_image_files=False)

		# Train the network
		cat_counts_sorted_by_val = train_mnist_or_att_faces(i * training_batch_size, train_set, network, cxt, 1)

		# Classify the test dataset with the trained network
		legend, row = test_mnist_or_att_faces(
			cat_counts_sorted_by_val, aif_scale, translation_sets,
			invert_images, crop_images_exclusion, center_images_exclusion, resampled_width, False, None,
			(i + 1) * training_batch_size, num_testing_images, test_set, network, cxt, 1
		)
		results_table.append(row)

	return legend, results_table


def run_mnist():
	if args.command_line_flag:
		# Run one configuration as specified on the command line.
		# This is particularly useful for multi-core, parallel data-gathering by running the script with various
		# configurations in multiple simultaneous terminals (shells).
		configs = [
			[
				False, args.crop_images_exclusion, args.center_images_exclusion, args.resampled_width, args.euclidean_norm,
				args.translations, args.num_training_images, args.num_testing_images, args.network_size,
			],
		]
	else:
		# Run a suite of configurations in a loop
		configs = [
			[
				False,  # Invert images
				None,  # Crop exclusion
				None,  # Center exclusion
				16,  # Resample size
				0,  # Euclidean distance norm, 0: don't use Euclidean (use L1 instead)
				False,  # Use translation sets
				10000,  # Train size (max 60,000)
				1000,   # Test size (max 10,000)
				0,  # Network size (see CM1KEmulator.__init__() for explanation)
			],
			[
				False,  # Invert images
				0,  # Crop exclusion
				None,  # Center exclusion
				16,  # Resample size
				0,  # Euclidean distance norm, 0: don't use Euclidean (use L1 instead)
				True,  # Use translation sets
				60000,  # Train size (max 60,000)
				10000,  # Test size (max 10,000)
				0,  # Network size (see CM1KEmulator.__init__() for explanation)
			],
			# Add more configurations as you see fit
		]

	results_table = []
	for config in configs:
		print "\n\n\n\n\nCONFIG: {}".format(config)
		legend, rows = run_oneconfig_mnist(
			config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8])
		assert len(legend) == len(rows[0])
		for ci, row in enumerate(rows):
			results_table.append(row)

	# See run_att_faces() to see how the results_table can be dumped to the screen for copy/paste to a spreadsheet.


def run_oneconfig_att_faces(invert_images, train_set_proportion, randomize_train_set, rseed):
	"""
	This test relies on the AT&T Faces database. You must supply paths to the files before this test can run.
	The paths are currently hard-coded into the corresponding source file.
	"""
	network = cm1k.CM1KEmulator(network_size=0)

	# See corresponding section in run_oneconfig_mnist() for explanation
	aif_scale = 1

	# See corresponding section in run_oneconfig_mnist() for explanation
	translation_sets = [
		((0, 0),),
		((0, -1), (0, 1), (-1, 0), (1, 0)),
		((-1, -1), (-1, 1), (1, -1), (1, 1)),
		((0, -2), (0, 2), (-2, 0), (2, 0)),
		((1, -2), (1, 2), (-2, 1), (2, 1), (-1, -2), (-1, 2), (-2, -1), (2, -1)),
	]

	cxt = 1

	# Read the entire dataset, then split into train and test subsets
	face_imgs = att_db_of_faces.read_att_db_of_faces(faces_data_dir, invert_images, print_images=False)
	train_set_cutoff = 0
	train_set = []
	test_set = []
	for lbl in face_imgs:
		one_face_imgs = face_imgs[lbl]
		assert(len(one_face_imgs) == 10)
		train_set_cutoff = int(round(10 * train_set_proportion))
		if lbl == 0:
			print "train_set_cutoff : {}".format(train_set_cutoff)
		for i in xrange(0, train_set_cutoff):
			train_set.append((one_face_imgs[i], lbl))
		for i in xrange(train_set_cutoff, 10):
			test_set.append((one_face_imgs[i], lbl))

	assert(len(train_set) == train_set_cutoff * 40)
	assert(len(test_set) == 400 - train_set_cutoff * 40)
	print "Size train and test sets: {}, {}".format(len(train_set), len(test_set))

	train_set_to_use = train_set

	# Optionally randomize the order of the train set.
	# The sets start out grouped by category, which could affect training perhaps.
	if randomize_train_set:
		rd.seed(rseed)

		train_set_randomized = []
		indices = range(0, len(train_set))
		rd.shuffle(indices)
		print "Randomized train set indices: {}".format(indices)

		for idx in indices:
			train_set_randomized.append(train_set[idx])

		train_set_to_use = train_set_randomized

	# Train the network
	cat_counts_sorted_by_val = train_mnist_or_att_faces(0, train_set_to_use, network, cxt, 1)

	# Classify the test dataset with the trained network
	legend, row = test_mnist_or_att_faces(
		cat_counts_sorted_by_val, aif_scale, translation_sets,
		invert_images, 0, 0, 16, randomize_train_set, rseed,
		len(train_set), len(test_set), test_set, network, cxt, 1
	)
	results_table = [row, ]

	return legend, results_table


def run_att_faces():
	# See run_mnist() for some ideas how to generalize the configuration command line arguments

	# Configurations are organized into two levels. The outer level specifies various parameters while the inner level
	# specifies a 'batch' of random-number-generator seeds. For a given outer config, a series of runs will be made that
	# differ only by random seed, with their results statistically aggregated into means and confidence intervals that
	# characterize the outer level config's performance.
	configs = []
	for train_proportion in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
		for rseed in xrange(0, 10):
			configs.append([
				False,  # Invert images
				train_proportion,  # Proportion of dataset used for training, the rest used for testing
				True,  # Randomize the dataset (its original order is highly organized and not well-suited for training)
				rseed,  # Random seed
			])

	legend = None
	results_table = []
	for config in configs:
		print "\n\n\n\n\nCONFIG: {}".format(config)
		legend, rows = run_oneconfig_att_faces(config[0], config[1], config[2], config[3])
		assert len(legend) == len(rows[0])
		assert len(rows) == 1
		for ci, row in enumerate(rows):
			results_table.append(row)

	# I don't like use DataFrames for screen output.  It's too hard to transfer elsewhere via copy/paste.
	# So, just produce a tilde-delimited data dump instead.  Why not TABs?  B/c some terminals don't insert a tab
	# character when the space to the next element is 0, which then breaks any copy/paste elsewhere (say a spreadsheet).
	# Alternatively, we could directly write a CSV file to disk here (via or not via DataFrames).
	print "################################################################################"
	print "FINAL RESULTS"
	print '~'.join(legend)
	for row in results_table:
		print '~'.join(row)

	# Here's the DataFrame approach
	# all_results_df = pd.DataFrame(results_table, rows=legend)
	# pd.options.display.max_rows = 999
	# pd.options.display.max_rows = 999
	# print all_results_df


def train_iris_or_mushroom(train_set, network, cxt, trace_mod, trace_test):
	print "################################################################################"
	print "TRAIN with {} instances".format(len(train_set))
	all_aifs = []
	for i, (features, lbl) in enumerate(train_set):
		# print "================================================================================"
		network.learn(features, lbl, cxt)

		# As training instances are incrementally presented, occasionally dump some ongoing metrics
		if i % trace_mod == trace_test or i == len(train_set) - 1:
			num_committed_neurons = network.read_ncount()
			num_degenerate = 0
			aifs = []
			for neuron in network.neurons:
				if neuron.state == nrn.NeuronState.com:
					if neuron.degenerate:
						num_degenerate += 1
					aifs.append(neuron.aif)
			assert(len(aifs) == num_committed_neurons)
			all_aifs.append(aifs)

			aif_min = np.min(aifs)
			aif_max = np.max(aifs)
			aif_mean = np.mean(aifs)
			aif_stddev = np.std(aifs)
			aif_median = np.median(aifs)
			aifs_hist = np.histogram(aifs)

			print "AIF summary:"
			print "{:>5}\t{:>5}\t{:>5}\t{:>5}\t{:>5}\t{:>10.1f}\t{:>10.1f}\t{:>10.1f}".format(
				i, num_committed_neurons, num_degenerate, aif_min, aif_max, aif_mean, aif_stddev, aif_median)

			print "AIF histogram:"
			print list(aifs_hist[0])
			print list(aifs_hist[1])

			print "AIFs:"
			print ' '.join([str(x) for x in sorted(aifs)])

			print "AIFs Dict:"
			aifs_dct = Counter()
			for aif in aifs:
				aifs_dct[aif] += 1
			for aif in sorted(aifs_dct.keys()):
				print "{}\t{}".format(aif, aifs_dct[aif])

			print "--------------------"

	# Dump the AIF distribution
	for aifs_bag in all_aifs:
		aifs_dct = Counter()
		for aif in aifs_bag:
			aifs_dct[aif] += 1
		row = ""
		for aif in sorted(aifs_dct.keys()):
			row += "{}\t{}\t".format(aif, aifs_dct[aif])
		print row


def test_iris_or_mushroom_out(
		max_aif, train_set_size, test_set_size, randomize_train_set, rseed, network_size,
		num_id0_unc0, num_id1_unc0, num_id0_unc1, num_id1_unc1, pos_mode_vote_sum, neg_mode_vote_sum,
		num_correct_id, num_correct_unc, num_wrong_id, num_wrong_unc
	):
	"""
	Dump the results of classification of a test set using a trained network
	"""

	#Calculate precision, recall, and F1 score
	correct_total = num_correct_id + num_correct_unc
	wrong_total = num_wrong_id + num_wrong_unc
	precision = float(correct_total) / float(correct_total + wrong_total) \
		if (correct_total + wrong_total) > 0 else 0
	recall = float(correct_total) / float(correct_total + num_id0_unc0) \
		if (correct_total + num_id0_unc0) > 0 else 0
	f1_eqtn1 = (2. * correct_total) / (2. * correct_total + wrong_total + num_id0_unc0) \
		if (correct_total + wrong_total + num_id0_unc0) > 0 else 0
	# f1_eqtn2 = 2. * ((precision * recall) / (precision + recall))

	# Dump all the results to the screen, with labels, for maximum legibility
	print "################################################################################"

	print "Results:"
	print "Max AIF:           {:>4}".format(max_aif)
	print "Train set size:    {:>4}".format(train_set_size)
	print "Test set size:     {:>4}".format(test_set_size)
	print "Randomize:         {:>4}".format(1 if randomize_train_set else 0)
	print "Random seed:       {:>4}".format(rseed)
	print "Num committed:     {:>4}".format(network_size)
	print "num_id0_unc0:      {:>4}".format(num_id0_unc0)
	print "num_id1_unc0:      {:>4}".format(num_id1_unc0)
	print "num_id0_unc1:      {:>4}".format(num_id0_unc1)
	print "num_id1_unc1:      {:>4}".format(num_id1_unc1)
	print "pos_mode_vote_sum: {:>4}".format(pos_mode_vote_sum)
	print "neg_mode_vote_sum: {:>4}".format(neg_mode_vote_sum)
	print "tot_mode_vote_sum: {:>4}".format(pos_mode_vote_sum + neg_mode_vote_sum)
	print "num correct total: {:>4}".format(correct_total)
	print "num_correct_id:    {:>4}".format(num_correct_id)
	print "num_correct_unc:   {:>4}".format(num_correct_unc)
	print "num wrong total:   {:>4}".format(wrong_total)
	print "num_wrong_id:      {:>4}".format(num_wrong_id)
	print "num_wrong_unc:     {:>4}".format(num_wrong_unc)
	print "precision:         {:>4}".format(precision)
	print "recall:            {:>4}".format(recall)
	print "F1:                {:>4}".format(f1_eqtn1)
	# print "F1:                {:>4}".format(f1_eqtn2)

	# Create a legend that can easily be copy/pasted into a spreadsheet
	legend = [
		'MAX AIF',
		'TRAIN SET SIZE',
		'TEST SET SIZE',
		'RANDOMIZE',
		'RANDOM SEED',
		'COMMITTED NEURONS',
		'NUM ID0 UNC0',
		'NUM ID1 UNC0',
		'NUM ID0 UNC1',
		'NUM ID1 UNC1',
		'POS MODE VOTE SUM',
		'NEG MODE VOTE SUM',
		'TOT MODE VOTE SUM',
		'NUM CORRECT TOTAL',
		'NUM CORRECT ID',
		'NUM CORRECT UNC',
		'NUM WRONG TOTAL',
		'NUM WRONG ID',
		'NUM WRONG UNC',
		'PRECISION',
		'RECALL',
		'F1',
		# 'F1',
	]

	# Create an output array that matches the legend (above) that can easily be copy/pasted into a spreadsheet
	output = []
	output.append("{}".format(max_aif))
	output.append("{}".format(train_set_size))
	output.append("{}".format(test_set_size))
	output.append("{}".format(1 if randomize_train_set else 0))
	output.append("{}".format(rseed))
	output.append("{}".format(network_size))
	output.append("{}".format(num_id0_unc0))
	output.append("{}".format(num_id1_unc0))
	output.append("{}".format(num_id0_unc1))
	output.append("{}".format(num_id1_unc1))
	output.append("{}".format(pos_mode_vote_sum))
	output.append("{}".format(neg_mode_vote_sum))
	output.append("{}".format(pos_mode_vote_sum + neg_mode_vote_sum))
	output.append("{}".format(correct_total))
	output.append("{}".format(num_correct_id))
	output.append("{}".format(num_correct_unc))
	output.append("{}".format(wrong_total))
	output.append("{}".format(num_wrong_id))
	output.append("{}".format(num_wrong_unc))
	output.append("{}".format(precision))
	output.append("{}".format(recall))
	output.append("{}".format(f1_eqtn1))
	# output.append("{}".format(f1_eqtn2))

	assert len(legend) == len(output)

	print "Legend ({}):".format(len(legend))
	print '\n'.join(legend)
	print "Results ({}):".format(len(output))
	print '\n'.join(output)

	return legend, output


def test_iris_or_mushroom(max_aif, train_set_size, test_set_size, randomize_train_set, rseed, test_set, network, cxt):
	"""
	Classify a test dataset with a trained network
	"""
	print "################################################################################"
	assert(len(test_set) == test_set_size)
	print "TEST with {} instances".format(test_set_size)
	num_id0_unc0 = 0
	num_id1_unc0 = 0
	num_id0_unc1 = 0
	num_id1_unc1 = 0
	num_correct_id = 0
	num_correct_unc = 0
	num_wrong_id = 0
	num_wrong_unc = 0

	trace_mod = 1
	if len(test_set) >= 100:
		trace_mod = 10
	if len(test_set) >= 1000:
		trace_mod = 100
	if len(test_set) >= 10000:
		trace_mod = 1000

	# See test_mnist_or_att_faces() for a description of these variables
	pos_mode_vote_sum = 0
	neg_mode_vote_sum = 0

	# Iterate over the test set
	for instance_idx, (features, lbl) in enumerate(test_set):
		# print "================================================================================"
		# print "Idx: {} {}".format(instance_idx, lbl)

		# Broadcast the image to the network and retrieve the classification results
		id_, unc, firing_neurons = network.broadcast(features, cxt, False)
		firing_neuron_cat_dists = [(x.cat, x.dist) for x in firing_neurons]
		best_neuron = firing_neurons[-1] if firing_neurons else None

		# print "  id, unc: {} {}".format(id_, unc)
		# print "  Firing neurons: {}".format(firing_neuron_cat_dists)

		assert(id_ == 0 or unc == 0)  # id and unc can't both be true
		if id_ != 0 or unc != 0:  # If id or unc is true, there must be a best neuron
			assert best_neuron
		if best_neuron:  # If there is a best neuron, id or unc must be true
			assert(id_ != 0 or unc != 0)

		if best_neuron:  # id_ != 0 or unc != 0:
			dist = network.read_dist()
			assert(dist == best_neuron.dist)

		# Get the set of all matched categories
		matched_cats = set([x[0] for x in firing_neuron_cat_dists])
		# print "  Uncertain, matched cats: {} {}".format(uncertain, matched_cats)

		# Get the mode classification (which category got the most hits)
		cat_mode = None
		cat_occurrences = Counter()
		for firing_neuron in firing_neuron_cat_dists:
			cat_occurrences[firing_neuron[0]] += 1
		cat_occurrences_sorted = cat_occurrences.most_common()
		if len(cat_occurrences_sorted) >= 2 and cat_occurrences_sorted[0][1] > cat_occurrences_sorted[1][1]:
			cat_mode = cat_occurrences_sorted[0][0]

		# There are three cases in this outer level condition:
		# (1) No neurons fired
		# (2) All firing neurons agree on the category
		# (3) Some firing neurons disagree on the category
		status = 0  # The id/unc correct/wrong status
		mode_effect = 0  # 1: cat_mode would correct a wrong classification, -1: it would wrong a correct one
		if len(matched_cats) == 0:
			# Unclassified: no neurons fired
			num_id0_unc0 += 1
			status = -1
		elif len(matched_cats) == 1:  # All firing neurons agree on the category
			num_id1_unc0 += 1
			if best_neuron.cat == lbl:   # Correct
				num_correct_id += 1
				status = 1
				if cat_mode is not None and cat_mode != lbl:
					assert False  # All firing neurons agreed and the best neuron was correct
			else:  # Wrong
				num_wrong_id += 1
				status = 2
				if cat_mode is not None and cat_mode == lbl:
					assert False  # All firing neurons agreed and the best neuron was wrong
		elif len(matched_cats) > 1:  # Some firing neurons disagree on the category
			num_id0_unc1 += 1
			if best_neuron.cat == lbl:   # Correct
				num_correct_unc += 1
				status = 11
				if cat_mode is not None and cat_mode != lbl:
					mode_effect = -1
			else:  # Wrong
				num_wrong_unc += 1
				status = 12
				if cat_mode is not None and cat_mode == lbl:
					mode_effect = 1

			# Keep track of the effect that mode-voting would have relative to winner-takes-all
			if mode_effect != 0:
				if mode_effect > 0:
					pos_mode_vote_sum += mode_effect
				else:
					neg_mode_vote_sum += mode_effect
				print "{:0>5}    lbl:{}    #cts:{}    ctm:{:_>4}    sts:{}    mef:{}".format(
					instance_idx, lbl, len(matched_cats), cat_mode, status, mode_effect)
				fn_out = "    "
				for cat_dist in reversed(firing_neuron_cat_dists):
					fn_out += "    {}".format(cat_dist)
				print fn_out

		# print "  Status: {}".format(status)

		# if instance_idx % trace_mod == 0:
		# 	print "{:>5}    {} {}    {:>2}    {}    {:40} {}".format(
		# 		instance_idx, id_, unc, status, lbl, firing_neuron_cat_dists, list(matched_cats))

	# Dump the results
	legend, output = test_iris_or_mushroom_out(
		max_aif, train_set_size, test_set_size, randomize_train_set, rseed, network.read_ncount(),
		num_id0_unc0, num_id1_unc0, num_id0_unc1, num_id1_unc1, pos_mode_vote_sum, neg_mode_vote_sum,
		num_correct_id, num_correct_unc, num_wrong_id, num_wrong_unc
	)

	return legend, output


def run_oneconfig_iris(max_aif, train_set_proportion, randomize_train_set, rseed):
	"""
	This test relies on the IRIS database. You must supply paths to the files before this test can run.
	The paths are currently hard-coded into the correspoding source file.
	"""
	network = cm1k.CM1KEmulator(network_size=0)
	network.write_maxif(max_aif)  # Override the default max AIF
	print "Max AIF: {}".format(network.read_maxif())

	iris_num_instances_per_cat = 50
	iris_num_categories = 3

	# Read the entire dataset
	data_grouped_by_cat = iris.read_iris(iris_data_dir)
	assert(len(data_grouped_by_cat) == 3)
	for cat in data_grouped_by_cat:
		assert(len(data_grouped_by_cat[cat]) == iris_num_instances_per_cat)

	# Split into train and test subsets
	num_train_items_per_cat = int(round(50 * train_set_proportion))
	print "Size train and set sets per category A: {}, {}".format(
		num_train_items_per_cat, iris_num_instances_per_cat - num_train_items_per_cat)
	train_set = []
	test_set = []
	for cat_idx, cat in enumerate(data_grouped_by_cat):
		data_one_cat = data_grouped_by_cat[cat]

		# The categories are initially strings. Convert them to ints by taking their index.
		for i in xrange(0, num_train_items_per_cat):
			train_set.append((data_one_cat[i], cat_idx))
		for i in xrange(num_train_items_per_cat, iris_num_instances_per_cat):
			test_set.append((data_one_cat[i], cat_idx))
	assert(len(train_set) == num_train_items_per_cat * 3)
	assert(len(test_set) == (iris_num_instances_per_cat * iris_num_categories) - num_train_items_per_cat * 3)
	print "Size train and set sets per category B: {}, {}".format(len(train_set), len(test_set))

	train_set_to_use = train_set

	# Optionally randomize the order of the train set (highly recommended).
	# The sets start out grouped by category, which could affect training perhaps.
	if randomize_train_set:
		rd.seed(rseed)

		train_set_randomized = []
		indices = range(0, len(train_set))
		rd.shuffle(indices)
		print "Randomized train set indices: {}".format(indices)

		for idx in indices:
			train_set_randomized.append(train_set[idx])

		train_set_to_use = train_set_randomized

	cxt = 1

	# Train the network
	train_iris_or_mushroom(train_set_to_use, network, cxt, 40, 39)

	# Classify the test dataset with the trained network
	legend, output = test_iris_or_mushroom(max_aif, len(train_set), len(test_set), randomize_train_set, rseed,
										   test_set, network, cxt)

	return legend, output


def run_iris():
	# See run_mnist() for some ideas how to use command line arguments to describe the configurations.
	# See run_att_faces() for a description of the 'config batches' shown below.
	config_batches = []

	if args.max_aif is not None:
		max_aif_range = [args.max_aif]
	else:
		max_aif_range = [128, 64, 32, 16, 8, 4, 2]

	for max_aif in max_aif_range:
		for train_proportion in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
			config_batch = []
			for rseed in xrange(0, 10):
				config_batch.append([max_aif, train_proportion, True, rseed])
			config_batches.append(config_batch)

	# Iterate over the configuration batches
	legend = None
	results_table = []
	means_table = []
	conf_int_95s_table = []
	for config_batch in config_batches:
		# Iterate over the random seeds of one batch and statistically aggregate the results
		results_one_config_table = []
		for config in config_batch:
			print "\n\n\n\n\n"
			print "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
			print "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
			print "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
			print "CONFIG: {}".format(config)
			legend, row = run_oneconfig_iris(config[0], config[1], config[2], config[3])
			assert len(legend) == len(row)
			results_table.append(row)
			row_f = [float(x) for x in row]
			for i in xrange(6, 16):
				row_f[i] /= float(row_f[2])
			results_one_config_table.append(row_f)
		means = np.mean(results_one_config_table, axis=0)
		stddevs = np.std(results_one_config_table, axis=0)
		means_one_config = []
		conf_int_95s_one_config = []
		for row in xrange(len(means)):
			mean = means[row]
			stddev = stddevs[row]
			stderr = stddev / math.sqrt(len(config_batch))
			conf_int_95 = stderr * 1.96
			means_one_config.append(mean)
			conf_int_95s_one_config.append(conf_int_95)
		means_table.append([str(x) for x in means_one_config])
		conf_int_95s_table.append([str(x) for x in conf_int_95s_one_config])

	# See run_att_faces() for an explanation and description of the final output
	print "################################################################################"
	print "FINAL RESULTS"
	print '~'.join(legend)
	for row in results_table:
		print '~'.join(row)

	print
	print '~'.join(legend)

	for i in xrange(0, len(means_table)):
		print '~'.join(means_table[i]) + '~' + '~'.join(conf_int_95s_table[i])

	# print "################################################################################"
	# print "FINAL RESULTS"
	# all_results_df = pd.DataFrame(results_table, rows=legend)
	# pd.options.display.max_rows = 999
	# pd.options.display.max_rows = 999
	# print all_results_df


def run_oneconfig_mushroom(max_aif, train_set_proportion, randomize_train_set, rseed):
	"""
	This test relies on the MUSHROOM database. You must supply paths to the files before this test can run.
	The paths are currently hard-coded into the correspoding source file.
	"""
	network = cm1k.CM1KEmulator(network_size=0)
	network.write_maxif(max_aif)  # Override the default max AIF
	print "Max AIF: {}".format(network.read_maxif())

	# Read the entire dataset
	data = mushroom.read_mushroom(mushroom_data_dir)
	assert(len(data) == 8124)

	# Split into train and test subsets
	train_set_cutoff = int(round(8124 * train_set_proportion))
	print "train_set_cutoff : {}".format(train_set_cutoff)
	train_set = []
	test_set = []
	for i in xrange(0, train_set_cutoff):
		train_set.append(data[i])
	for i in xrange(train_set_cutoff, 8124):
		test_set.append(data[i])

	assert(len(train_set) == train_set_cutoff)
	assert(len(test_set) == 8124 - train_set_cutoff)
	print "Size train and set sets: {}, {}".format(len(train_set), len(test_set))

	train_set_to_use = train_set

	# Optionally randomize the order of the train set (highly recommended).
	# The sets start out grouped by category, which could affect training perhaps.
	if randomize_train_set:
		rd.seed(rseed)

		train_set_randomized = []
		indices = range(0, len(train_set))
		rd.shuffle(indices)
		# print "Randomized train set indices: {}".format(indices)

		for idx in indices:
			train_set_randomized.append(train_set[idx])

		train_set_to_use = train_set_randomized

	cxt = 1

	# Train the network
	train_iris_or_mushroom(train_set_to_use, network, cxt, 1000, 0)

	# Classify the test dataset with the trained network
	legend, output = test_iris_or_mushroom(max_aif, len(train_set), len(test_set), randomize_train_set, rseed,
										   test_set, network, cxt)

	return legend, output


def run_mushroom():
	# See run_mnist() for some ideas how to use command line arguments to describe the configurations.
	# See run_att_faces() for a description of the 'config batches' shown below.
	config_batches = []

	if args.max_aif is not None:
		max_aif_range = [args.max_aif]
	else:
		max_aif_range = [64, 32, 16, 8, 6, 4]

	for max_aif in max_aif_range:
		for train_proportion in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
			config_batch = []
			for rseed in xrange(0, 10):
				config_batch.append([max_aif, train_proportion, True, rseed])
			config_batches.append(config_batch)

	# Iterate over the configuration batches
	legend = None
	results_table = []
	means_table = []
	conf_int_95s_table = []
	for config_batch in config_batches:
		# Iterate over the random seeds of one batch and statistically aggregate the results
		results_one_config_table = []
		for config in config_batch:
			print "\n\n\n\n\n"
			print "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
			print "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
			print "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
			print "CONFIG: {}".format(config)
			legend, row = run_oneconfig_mushroom(config[0], config[1], config[2], config[3])
			assert len(legend) == len(row)
			results_table.append(row)
			row_f = [float(x) for x in row]
			for i in xrange(6, 16):
				row_f[i] /= float(row_f[2])
			results_one_config_table.append(row_f)
		means = np.mean(results_one_config_table, axis=0)
		stddevs = np.std(results_one_config_table, axis=0)
		means_one_config = []
		conf_int_95s_one_config = []
		for row in xrange(len(means)):
			mean = means[row]
			stddev = stddevs[row]
			stderr = stddev / math.sqrt(len(config_batch))
			conf_int_95 = stderr * 1.96
			means_one_config.append(mean)
			conf_int_95s_one_config.append(conf_int_95)
		means_table.append([str(x) for x in means_one_config])
		conf_int_95s_table.append([str(x) for x in conf_int_95s_one_config])

	# See run_att_faces() for an explanation and description of the final output
	print "################################################################################"
	print "FINAL RESULTS"
	print '~'.join(legend)
	for row in results_table:
		print '~'.join(row)

	print
	print '~'.join(legend)

	for i in xrange(0, len(means_table)):
		print '~'.join(means_table[i]) + '~' + '~'.join(conf_int_95s_table[i])

	# print "################################################################################"
	# print "FINAL RESULTS"
	# all_results_df = pd.DataFrame(results_table, rows=legend)
	# pd.options.display.max_rows = 999
	# pd.options.display.max_rows = 999
	# print all_results_df


def main():
	# Pick one (or more) of the following. I haven't bothered configuring a switch over these via run-time arguments yet.

	# UNIT TESTS
	run_unit_test_01(1)
	run_unit_test_02(1)
	run_unit_test_03(1)
	# run_unit_test_mnist_01(1)

	# POPULAR PUBLIC DATASETS
	# run_mnist()
	# run_att_faces()
	# run_iris()
	# run_mushroom()


if __name__ == "__main__":
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("-clf", "--command_line_flag", default=False, action='store_true')
	arg_parser.add_argument("-cre", "--crop_images_exclusion", type=int, default=None)
	arg_parser.add_argument("-cte", "--center_images_exclusion", type=int, default=None)
	arg_parser.add_argument("-wth", "--resampled_width", type=int)
	arg_parser.add_argument("-euc", "--euclidean_norm", type=int)
	arg_parser.add_argument("-trl", "--translations", default=False, action='store_true')
	arg_parser.add_argument("-xif", "--max_aif", type=int, default=None)
	arg_parser.add_argument("-trn", "--num_training_images", type=int)
	arg_parser.add_argument("-tst", "--num_testing_images", type=int)
	arg_parser.add_argument("-nws", "--network_size", type=int)

	# args will be globally accessible
	args = arg_parser.parse_args()

	main()
