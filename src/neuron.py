"""
Keith Wiley
kwiley@keithwiley.com
http://keithwiley.com
"""

from enum import Enum
import math as math
import log
import cm1k_emulator as cm1k


class NeuronState(Enum):
	"""
	CM1K neuron state
	"""
	idle = 0  # Unused
	rtl = 1  # Ready to learn
	com = 2  # Committed


class Neuron:
	def __init__(self, id_, chip):
		"""
		cxt:
		The CM1K offer 127 contexts in the range 1-127.
		Context 0 is used during training to train all neurons against an input regardless of their contexts.

		cat:
		The CM1K offers 32767 categories in the range 1-32767.
		Category 0 is used during training to present counterexamples (to shrink the neurons' AIFs)
		If the neuron degenerates, bit 15 of the category is set to 1
		(i.e., 32768 will be added to the category).

		aif:
		This should be indicated in the same range as dist, below, as determined by norm.

		dist:
		If the norm is L1, then distances will be in the range 0-65280 (255 x 256).
		If the norm is Lsup (i.e., max), then distances will be in the range 0-255.

		pattern:
		A byte array which will be compared on a byte-by-byte basis (not bit-by-bit, so not hamming distance).
		"""
		log.trace("Neuron.init()")

		self.id_ = id_
		self.chip = chip
		self.state = NeuronState.idle
		self.cxt = 0  # Context
		self.cat = 0  # Category
		self.aif = 0  # Active influence field
		self.degenerate = False  # True when aif shrinks to minif
		self.dist = 0
		self.pattern = []  # Components (the pattern or rbf "center" stored in this neuron)

	def __repr__(self):
		# Include the pattern
		# return "id{} st{} cxt{} cat{} aif{} deg{} dst{} {}".format(
		# 	self.id_, self.state, self.cxt, self.cat, self.aif, self.degenerate, self.dist, self.pattern)
		# Exclude the pattern
		return "id{} st{} cxt{} cat{} aif{} deg{} dst{}".format(
			self.id_, self.state, self.cxt, self.cat, self.aif, self.degenerate, self.dist)

	def update_dist(self, indexcomp, comp, norm, last_comp=False, classifier=None):
		"""
		Calculate the distance from the supplied pattern to the stored pattern
		"""
		# Called from CM1KEmulator.update_all_neuron_dists(), i.e., whenever COMP or LCOMP is updated
		log.trace("Neuron.update_dist()")

		if norm == cm1k.CM1KDistNorm.l1:
			self.dist += abs(comp - self.pattern[indexcomp])
		elif norm == cm1k.CM1KDistNorm.lsup:
			self.dist = max(abs(comp - self.pattern[indexcomp]), self.dist)
		elif norm == cm1k.CM1KDistNorm.euc:
			self.dist += (comp - self.pattern[indexcomp]) * (comp - self.pattern[indexcomp])

		if last_comp:
			if norm == cm1k.CM1KDistNorm.euc:
				self.dist = int(round(math.sqrt(self.dist)))
			if (classifier == cm1k.CM1KClassifier.rbf and self.dist < self.aif) or classifier == cm1k.CM1KClassifier.knn:
				# The neuron has fired
				self.chip.store_firing_neuron(self)

	def reset_dist(self):
		"""
		Reset the distance to 0
		"""
		# Called from CM1KEmulator.reset_indexcomp(), i.e., whenever LCOMP is updated
		log.trace("Neuron.reset_dist()")

		self.dist = 0

	def broadcast(self, input_, norm, classifier=None, aif_scale=1):
		"""
		Used for high level broadcast, in which the input is processed in bulk instead of per-component, i.e., per byte.
		input_ of len 1-256 (for proper CM1K emulation, otherwise unlimited)
		norm: A DistNorm enum
		classifier: A Classifier enum
		aif_scale: Modify the aif when determining whether the fire. The aif can also be permanently scaled via
		CM1KEmulator.scale_all_aifs(), but this parameter enables the same behavior without altering the neuron.
		"""
		# Called from CM1KEmulator.broadcast()
		log.trace("Neuron.broadcast()")

		# This shouldn't be necessary. This function should only be called on committed and the rtl neurons.
		if self.state == NeuronState.idle:
			log.error("Neuron.broadcast() called on idle neuron")
			return

		self.dist = 0  # NOTE: Not sure this is necessary. Also, undecided whether this should simply call reset_dist().
		if norm == cm1k.CM1KDistNorm.l1:
			for i, comp in enumerate(input_):
				self.dist += abs(comp - self.pattern[i])
		elif norm == cm1k.CM1KDistNorm.lsup:
			for i, comp in enumerate(input_):
				self.dist = max(abs(comp - self.pattern[i]), self.dist)
		elif norm == cm1k.CM1KDistNorm.euc:
			for i, comp in enumerate(input_):
				self.dist += (comp - self.pattern[i]) * (comp - self.pattern[i])
			self.dist = int(round(math.sqrt(self.dist)))
		log.log("Single neuron cat{} dist: {:>5} < {:>5} ?".format(self.cat, self.dist, self.aif))

		# TODO: Use the minimum and maximum AIFs of each neuron (i.e., of each context)
		aif = self.aif if aif_scale == 1 else min(max(int(round(self.aif * aif_scale)), 0), 0xFFFF)

		if (classifier == cm1k.CM1KClassifier.rbf and self.dist < aif) or classifier == cm1k.CM1KClassifier.knn:
			# The neuron has fired
			log.log("Fire with dist{} aif{} cat{}".format(self.dist, aif, self.cat))
			self.chip.store_firing_neuron(self)

	def commit(self, cxt, cat, aif, pattern):
		"""
		Commit this neuron to the network. It will already have received a new pattern in the immediately preceding broadcast.
		"""
		log.trace("Neuron.commit() cxt{} cat{} aif{}".format(cxt, cat, aif))

		self.state = NeuronState.com
		self.cxt = cxt
		self.cat = cat
		self.aif = aif
		self.dist = 0
		# We shouldn't need to assign the pattern. It should already be assigned, but no harm done.
		self.pattern = pattern

	def shrink_aif(self, new_aif, minif):
		"""
		Shrink the AIF to the supplied AIF, but do not shrink below the indicate minimum AIF.
		"""
		log.trace("Neuron.shrink_aif()")

		if new_aif >= self.aif:
			# TODO: create unit test where misfiring neuron has exactly the same distance as the best neuron.
			raise ValueError("Attempted to shrink AIF ({}) to higher AIF ({})".format(self.aif, new_aif))

		self.aif = new_aif
		if self.aif <= minif:
			self.aif = minif
			self.degenerate = True
			log.log("This neuron has degenerated")

	def shrink_if_necessary(self, cat, new_aif, minif):
		"""
		Shrink if the AIF if categories don't match and error-compensating AIF < currently held AIF.
		"""
		log.trace("Neuron.shrink_if_necessary()")

		# TODO: create unit test where misfiring neuron has exactly the same distance as the best neuron.
		if cat != self.cat and new_aif < self.aif:
			self.shrink_aif(new_aif, minif)
