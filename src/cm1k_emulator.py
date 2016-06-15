from enum import Enum
from collections import OrderedDict
import log
import neuron as nrn

"""
The primary reference used to create this emulator was:
CM1K Hardware User's Manual
by General Vision
http://general-vision.com/documentation/
Although I designed the emulator using v3.4.8 of the manual, General Vision released an updated version during my work.
Consequently, I have updated the various page references in this file to reflect v4.0.2 of the manual.
"""


class CM1KMode(Enum):
	"""
	CM1K mode, either normal (for broadcasting and learning)
	or save-restore (for saving the entire network state after training, or for loading a saved state back).
	"""
	normal = 0
	save_restore = 1


class CM1KDistNorm(Enum):
	"""
	CM1K distance norm, either L1 or Lsup.
	Euclidean is also offered for comparison purposes, but the CM1K does not offer it.
	"""
	l1 = 0  # Manhattan distance between vectors (as sum of unsigned byte component differences)
	lsup = 1  # Max single byte component difference
	euc = 2  # Euclidean distance is not supported by the CM1K chip, but is included in the emulator for comparison


class CM1KClassifier(Enum):
	"""
	CM1K classifier, either radial basis function or k-nearest neighbors
	"""
	rbf = 0
	knn = 1


class CM1KEmulator:
	"""
	This class attempts to reproduce the NeuroMem API and emulate the CM1K chip.
	However, it also simply presents a basic RBF network, aside from CM1K specifics.

	To conform with the CM1K design, num_neurons should be a multiple of 1024, but this is not required by the emulator.
	To assist in this regard, a negative value passed into num_neurons will indicate the number of
	CM1K chips to use (i.e., it will be multiplied by 1024 to produce the actual neuron count).
	"""
	def __init__(self, network_size=0):
		"""
		network_size < 0: indicates the (inverse of the) number of CM1K chips (1024 neurons each).
		network_size > 0: indicates the number of neurons
							(only multiples of 1024 emulate a CM1K environment, but any value is permitted).
		network_size == 0: indicates unlimited neurons
							(the network will indefinitely allocate new neurons as training requires them).
		"""
		log.trace("CM1KEmulator.init(): {}".format(network_size))

		# Create the neurons (or if unlimited, then create the first neuron)
		self.unlimited_neurons = (network_size == 0)
		if network_size < 0:
			num_neurons = -network_size * 1024
		elif network_size == 0:
			num_neurons = 1
		else:  # elif network_size > 0:
			num_neurons = network_size
		self.neurons = []
		for i in xrange(0, num_neurons):
			self.neurons.append(nrn.Neuron(i, self))

		# Put the first neuron in ready-to-learn state
		self.neurons[0].state = nrn.NeuronState.rtl

		# A list of all firing neurons, sorted by distance
		self.firing_neurons = []

		# The input received by a broadcast
		self.input_ = []

		# The dist norm register is binary, so it can only indicate the CM1K's two official dist norms. In order to test
		# dist norms the CM1K doesn't support (e.g., Euclidean), we need a flag outside the registers as an override.
		self.euclidean_norm = False

		# CM1K Hardware Manual, p. 8
		self.cmd_ctrl_lines = [
			'DS', 'RW_', 'REG', 'DATA', 'RDY', 'ID_', 'UNC_'
		]

		# CM1K Hardware Manual, p. 10-14
		self.register_legend = OrderedDict()  # Register hardware defaults
		# Name, addr, default-value-or-na, normal-mode-RW, sr-mode-RW
		self.register_legend['NCR'] =        (0x00, 0x0001)  # -  RW
		self.register_legend['COMP'] =       (0x01, 0x0000)  # W  RW
		self.register_legend['LCOMP'] =      (0x02, -1)      # W  -
		# It is crucial that INDEXCOMP be assigned before DIST since they share the same address
		self.register_legend['INDEXCOMP'] =  (0x03, 0x0000)  # W  W   Shares register with DIST
		self.register_legend['DIST'] =       (0x03, 0xFFFF)  # R  R   Shares register with INDEXCOMP
		self.register_legend['CAT'] =        (0x04, 0xFFFF)  # RW RW
		self.register_legend['AIF'] =        (0x05, 0x4000)  # -  RW
		self.register_legend['MINIF'] =      (0x06, 0x0002)  # RW RW
		self.register_legend['MAXIF'] =      (0x07, 0x4000)  # RW -
		self.register_legend["TESTCOMP"] =   (0x08, -1)      # -  W   Not really needed by the emulator
		self.register_legend["TESTCAT"] =    (0x09, -1)      # -  W   Not really needed by the emulator
		self.register_legend['NID'] =        (0x0A, 0x0000)  # R  R
		self.register_legend['GCR'] =        (0x0B, 0x0001)  # RW -
		self.register_legend['RESETCHAIN'] = (0x0C, -1)      # -  W
		self.register_legend['NSR'] =        (0x0D, 0x0000)  # RW W
		self.register_legend['POWERSAVE'] =  (0x0E, -1)      # W  -   Not really needed by the emulator
		# It is crucial that NCOUNT be assigned before FORGET since they share the same address
		self.register_legend['NCOUNT'] =     (0x0F, 0x0000)  # R  R   Shares register with FORGET
		self.register_legend['FORGET'] =     (0x0F, -1)      # W  -   Shares register with NCOUNT

		# Initialize the registers with the hardware defaults
		self.registers = OrderedDict()
		for key, val in self.register_legend.iteritems():
			if val[0] not in self.registers:  # There are a few repeats, so just skip them
				self.registers[val[0]] = val[1]

		# Although INDEXCOMP is described as associated with register 3, it appears to only be associated by write,
		# not read. The actual value is not stored in register 3 (which stores DIST), so we must store it separately.
		self.indexcomp = 0

		log.log("New RBFNetwork created")
		self.dump_registers()

	# =========================================================================================================

	def __repr__(self):
		# Until otherwise needed...
		return "CM1K"

	def dump_registers(self):
		"""
		Log the register values
		"""
		for key, val in self.register_legend.iteritems():
			log.log("{:12} {:>2}: {:>10} {:>10}".format(
				key, val[0], self.registers[val[0]], "0x{:X}".format(self.registers[val[0]])))

	# =========================================================================================================

	def set_register_bit(self, register, bit):
		self.registers[self.register_legend[register][0]] |= (1 << bit)

	def clear_register_bit(self, register, bit):
		self.registers[self.register_legend[register][0]] &= ~(1 << bit)

	def toggle_register_bit(self, register, bit):
		self.registers[self.register_legend[register][0]] ^= 1 << bit

	def assign_register_bit(self, register, bit, val):
		self.registers[self.register_legend[register][0]] ^= \
			(-val ^ self.registers[self.register_legend[register][0]]) & (1 << bit)

	def get_register_bits(self, register, low_bit_pos, num_bits=1):
		return (self.registers[self.register_legend[register][0]] >> low_bit_pos) & ((1 << num_bits) - 1)

	# =========================================================================================================

	# Most of the crucial details of how to process the registers were taken from the CM1K Hardware Manual, p. 10-15

	# Naming convention: functions whose name ends in "non_ui" do not represent exposed functions of the NeuroMem API.
	# Rather, they offer internal accessor functions used by the emulator to write and read the various registers.

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# NCR register

	def write_ncr_nid(self, neuron_id_third_byte):
		# 0-255
		# The third byte of the neuron id. Only necessary if there are > 65535 neurons.
		# Generally, this method should only be called by write_total_ncount_non_ui().
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			raise ValueError("Can't write NCR in normal mode")
			# Although NCR is unwritable in normal mode, here is how it would be done:
			# self.registers[self.register_legend['NCR'][0]] &= 0
			# self.registers[self.register_legend['NCR'][0]] |= neuron_id_third_byte
		else:  # elif mode == Mode.save_restore:
			self.registers[self.register_legend['NCR'][0]] &= 255  # Clear high byte, preserve low byte
			self.registers[self.register_legend['NCR'][0]] |= neuron_id_third_byte << 8  # Assign high byte

	def read_ncr_nid(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			raise ValueError("Can't read NCR in normal mode")
			# Although NCR can't be read in normal mode, here is how it would be done:
			# return self.get_register_bits('NCR', 0, 8)
		else:  # elif mode == Mode.save_restore:
			return self.get_register_bits('NCR', 8, 8)

	# ----------------------------------------

	def write_ncr_norm(self, norm):
		# A DistNorm enum
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			raise ValueError("Can't write NCR in normal mode")
		else:  # elif mode == Mode.save_restore:
			if norm == CM1KDistNorm.l1:
				self.clear_register_bit('NCR', 7)
			else:  # if norm == DistNorm.lsup:
				self.set_register_bit('NCR', 7)

	def read_ncr_norm(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			# raise ValueError("Can't read NCR in normal mode")  # CM1K Hardware Manual, p. 14
			# cm1k_emulator.py", line 193
			return self.get_register_bits('NCR', 7)
		else:  # elif mode == Mode.save_restore:
			return self.get_register_bits('NCR', 7)

	def read_ncr_norm_enum(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			# raise ValueError("Can't read NCR DIST NORM in normal mode")  # CM1K Hardware Manual, p. 14
			# cm1k_emulator.py", line 704
			return CM1KDistNorm.l1 if self.read_ncr_norm() == 0 else CM1KDistNorm.lsup
		else:  # elif mode == Mode.save_restore:
			if self.euclidean_norm:
				return CM1KDistNorm.euc
			return CM1KDistNorm.l1 if self.read_ncr_norm() == 0 else CM1KDistNorm.lsup

	# ----------------------------------------

	def write_ncr_context(self, context):
		# 0-127
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			raise ValueError("Can't write NCR in normal mode")
		else:  # elif mode == Mode.save_restore:
			self.registers[self.register_legend['NCR'][0]] &= 65408  # Clear lower 7 bits, preserve high 9 bits
			self.registers[self.register_legend['NCR'][0]] |= context

	def read_ncr_context(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			raise ValueError("Can't read NCR in normal mode")
		else:  # elif mode == Mode.save_restore:
			return self.get_register_bits('NCR', 0, 7)

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# COMP register

	def write_comp(self, comp):
		# 0-255 (a one byte component of the 1 to 255 bytes input vector)
		# mode = self.read_nsr_mode_non_ui()
		self.registers[self.register_legend['COMP'][0]] |= comp
		self.update_all_neuron_dists()
		self.increment_indexcomp()

	def read_comp(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			raise ValueError("Can't read COMP in normal mode")
		else:  # elif mode == Mode.save_restore:
			comp = self.get_register_bits('COMP', 0, 8)
			self.increment_indexcomp()
			return comp

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# LCOMP register

	def write_lcomp(self, comp):
		# 0-255 (a one byte component of the 1 to 255 bytes input vector)
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.save_restore:
			raise ValueError("Can't write LCOMP in save_restore mode")
		self.registers[self.register_legend['LCOMP'][0]] |= comp
		self.firing_neurons = []
		self.update_all_neuron_dists(last_comp=True)
		self.reset_indexcomp()
		self.write_indexcomp(0)

	def read_lcomp(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			raise ValueError("Can't read LCOMP in normal mode")
		else:  # elif mode == Mode.save_restore:
			raise ValueError("Can't read LCOMP in save_restore mode")
		# return self.get_register_bits('LCOMP', 0, 8)

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# INDEXCOMP register

	def write_indexcomp(self, indexcomp):
		# 0-255 (the index of the next input vector component, which is at most 256 bytes long)
		# mode = self.read_nsr_mode_non_ui()

		# NOTE: I don't believe we can write the INDEXCOMP register directly, since it holds DIST
		# self.registers[self.register_legend['INDEXCOMP'][0]] |= indexcomp

		self.indexcomp = indexcomp

	def reset_indexcomp(self):
		self.indexcomp = 0

		# The docs aren't explicit that the neurons should reset their distance, but it seems to me they must!
		for neuron in self.neurons:
			neuron.reset_dist()

	def increment_indexcomp(self):
		# QSTN: What is the correct behavior if indexcomp is incremented past 255? The register supports two bytes.
		self.indexcomp += 1

	# There is no getter for INDEXCOMP. Reading the associated register corresponds to reading DIST.

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# DIST register

	# There is no UI setter for DIST. Writing the associated register corresponds to writing INDEXCOMP.
	# However we need an internal method for the CM1K and its neurons to fill the DIST register.
	def write_dist_non_ui(self, dist):
		# If dist norm is L1, then 0-65280. If dist norm is LSUP, then 0-255.
		self.registers[self.register_legend['DIST'][0]] &= 0
		self.registers[self.register_legend['DIST'][0]] |= dist

	def read_dist(self):
		# mode = self.read_nsr_mode_non_ui()
		self.update_firing_dist_and_cat()
		dist = self.registers[self.register_legend['DIST'][0]]
		return dist

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# CAT register

	def write_cat(self, cat):
		# 0-32767 (0 is not a true category, but rather used to present training counterexamples).
		# mode = self.read_nsr_mode_non_ui()
		self.registers[self.register_legend['CAT'][0]] &= 0
		self.registers[self.register_legend['CAT'][0]] |= cat
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.save_restore:
			# Move to the next neuron in the chain
			# TODO: Not done yet
			pass

	def read_cat(self):
		mode = self.read_nsr_mode_non_ui()
		cat = self.get_register_bits('CAT', 0, 15)
		if mode == CM1KMode.save_restore:
			# Move to the next neuron in the chain
			# TODO: Not done yet
			pass
		return cat

	def write_cat_degenerate(self, degenerate):
		# Boolean
		# mode = self.read_nsr_mode_non_ui()
		self.assign_register_bit('CAT', 15, 1 if degenerate else 0)

	def read_cat_degenerate(self):
		# mode = self.read_nsr_mode_non_ui()
		return self.get_register_bits('CAT', 15, 1)

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# AIF register

	def write_aif(self, aif):
		# If dist norm is L1, then 0-65280. If dist norm is LSUP, then 0-255.
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			raise ValueError("Can't write AIF in normal mode")
		else:  # elif mode == Mode.save_restore:
			self.registers[self.register_legend['AIF'][0]] |= aif

	def read_aif(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			raise ValueError("Can't read AIF in normal mode")
		else:  # elif mode == Mode.save_restore:
			return self.get_register_bits('AIF', 0, 16)

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# MINIF register

	def write_minif(self, minif):
		# If dist norm is L1, then 0-65280. If dist norm is LSUP, then 0-255.
		# mode = self.read_nsr_mode_non_ui()
		self.registers[self.register_legend['MINIF'][0]] = minif

	def read_minif(self):
		# mode = self.read_nsr_mode_non_ui()
		return self.registers[self.register_legend['MINIF'][0]]

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# MAXIF register

	def write_maxif(self, maxif):
		# If dist norm is L1, then 0-65280. If dist norm is LSUP, then 0-255.
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.save_restore:
			raise ValueError("Can't write MAXIF in SR mode")
		self.registers[self.register_legend['MAXIF'][0]] = maxif

	def read_maxif(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.save_restore:
			raise ValueError("Can't read MAXIF in SR mode")
		return self.registers[self.register_legend['MAXIF'][0]]

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# TESTCOMP register

	def write_testcomp(self):
		# Not implemented yet, not really needed by the emulator
		pass

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# TESTCAT register

	def write_testcat(self):
		# Not implemented yet, not really needed by the emulator
		pass

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# NID register

	# There is only a nonUI function for writing NID since the manual p. 14 indicates the register as unwritable.
	def write_nid_non_ui(self, nid):
		# 0-65535. If there are > 65535 neurons, the third id byte is in the high byte of NCR.
		# mode = self.read_nsr_mode_non_ui()
		self.registers[self.register_legend['NID'][0]] = nid

	def write_total_nid_non_ui(self, nid):
		# This method encapsulates both writing the lower two bytes to NID and writing the third byte to NCR.
		# First, write the lower two bytes to NID
		self.write_nid_non_ui(nid & 65535)
		# Second, write a third byte to the high byte of NCR
		self.write_ncr_nid(nid >> 16)

	def read_nid(self):
		# mode = self.read_nsr_mode_non_ui()
		return self.registers[self.register_legend['NID'][0]]

	def read_total_nid_non_ui(self):
		two_bytes = self.read_nid()
		if two_bytes == 0xFFFF:
			third_byte = self.read_ncr_nid()
			return third_byte << 16 | two_bytes
		return two_bytes

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# GCR register

	# ----------------------------------------

	def write_gcr_ncount(self, ncount_third_byte):
		# 0-127
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			self.registers[self.register_legend['GCR'][0]] &= 255  # Clear high byte, preserve low byte
			self.registers[self.register_legend['GCR'][0]] |= ncount_third_byte << 8  # Assign high byte
		if mode == CM1KMode.save_restore:
			raise ValueError("Can't write GCR in SR mode")

	def read_gcr_ncount(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			return self.get_register_bits('GCR', 8, 8)
		if mode == CM1KMode.save_restore:
			raise ValueError("Can't read GCR in SR mode")

	# ----------------------------------------

	def set_gcr_distnorm(self, norm):
		# A DistNorm value
		# mode = self.read_nsr_mode_non_ui()
		if norm == CM1KDistNorm.l1:
			self.clear_register_bit('GCR', 7)
		else:  # if norm == DistNorm.lsup:
			self.set_register_bit('GCR', 7)

	def read_gcr_distnorm(self):
		# mode = self.read_nsr_mode_non_ui()
		return self.get_register_bits('GCR', 7)

	def read_gcr_distnorm_enum(self):
		return CM1KDistNorm.l1 if self.read_gcr_distnorm() == 0 else CM1KDistNorm.lsup

	# ----------------------------------------

	def write_gcr_context(self, context):
		# 0-127
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			self.registers[self.register_legend['GCR'][0]] &= ~127  # Clear low 7 bits
			self.registers[self.register_legend['GCR'][0]] |= context
		if mode == CM1KMode.save_restore:
			raise ValueError("Can't write GCR in SR mode")

	def read_gcr_context(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			return self.registers[self.register_legend['GCR'][0]] & 127
		if mode == CM1KMode.save_restore:
			raise ValueError("Can't read GCR in SR mode")

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# RESETCHAIN register

	def write_resetchain(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			raise ValueError("Can't write RESETCHAIN in normal mode")
		else:  # elif mode == Mode.save_restore:
			ncount = self.read_ncount()
			self.write_ncount_non_ui(0 if ncount == 0 else 1)

	# There is no way to read the RESETCHAIN register. Doing so is a meaningless concept.

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# NSR register

	def set_nsr_unc(self):
		# mode = self.read_nsr_mode_non_ui()
		self.set_register_bit('NSR', 2)

	def clear_nsr_unc(self):
		# mode = self.read_nsr_mode_non_ui()
		self.clear_register_bit('NSR', 2)

	def read_nsr_unc(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			return self.get_register_bits('NSR', 2)
		else:  # elif mode == Mode.save_restore:
			raise ValueError("Can't read NSR in SR mode")

	# ----------------------------------------

	def set_nsr_id(self):
		# mode = self.read_nsr_mode_non_ui()
		self.set_register_bit('NSR', 3)

	def clear_nsr_id(self):
		# mode = self.read_nsr_mode_non_ui()
		self.clear_register_bit('NSR', 3)

	def read_nsr_id(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			return self.get_register_bits('NSR', 3)
		else:  # elif mode == Mode.save_restore:
			raise ValueError("Can't write NSR in SR mode")

	# ----------------------------------------

	def write_nsr_normal_mode(self):
		# mode = self.read_nsr_mode_non_ui()
		self.clear_register_bit('NSR', 4)

	def write_nsr_sr_mode(self):
		# Save-and-restore mode, for reading and writing the entire network structure (archival)
		# mode = self.read_nsr_mode_non_ui()
		self.set_register_bit('NSR', 4)

	def read_nsr_mode_non_ui(self):
		return self.get_register_bits('NSR', 4)

	def read_nsr_mode(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			return self.get_register_bits('NSR', 4)
		else:  # elif mode == Mode.save_restore:
			raise ValueError("Can't read NSR in SR mode")

	def read_nsr_mode_enum(self):
		return CM1KMode.normal if self.read_nsr_mode_non_ui() == 0 else CM1KMode.save_restore

	# ----------------------------------------

	def write_nsr_rbf_classifier(self):
		# mode = self.read_nsr_mode_non_ui()
		self.clear_register_bit('NSR', 5)

	def write_nsr_knn_classifier(self):
		# mode = self.read_nsr_mode_non_ui()
		self.set_register_bit('NSR', 5)

	def read_nsr_classifier(self):
		mode = self.read_nsr_mode_non_ui()
		if mode == CM1KMode.normal:
			return self.get_register_bits('NSR', 5)
		else:  # elif mode == Mode.save_restore:
			raise ValueError("Can't read NSR in SR mode")

	def read_nsr_classifier_enum(self):
		return CM1KClassifier.rbf if self.read_nsr_classifier() == 0 else CM1KClassifier.knn

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# POWERSAVE register

	def write_powersave(self):
		# Not implemented yet, not really needed by the emulator
		pass

	# There is no way to read the POWERSAVE register. Doing so is a meaningless concept.

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# FORGET register

	def write_forget(self, context):
		pass

	# There is no getter for FORGET. Reading the associated register corresponds to reading NCOUNT.

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# NCOUNT register

	# There is no UI setter for NCOUNT. Writing the associated register corresponds to writing FORGET.
	# However we need an internal method for the CM1K to fill the NCOUNT register.
	def write_ncount_non_ui(self, ncount):
		# 0-65535
		# Generally, this method should only be called from write_total_ncount_non_ui().
		self.registers[self.register_legend['NCOUNT'][0]] = ncount

	def write_total_ncount_non_ui(self, ncount):
		# 0-16,777,215
		# First, write the lower two bytes to NCOUNT
		self.write_ncount_non_ui(ncount & 65535)
		# Second, write a third byte to the high byte of GCR.
		# I am convinced the manual is full of errors, despite having gone through numerous revisions thus far.
		# P. 11 describes NCR has holding the third byte of RTL (aka ncount), but p. 12 and 14 describe NCR has holding
		# the third byte of NID. Furthermore, p. 10 and 14 describe GCR has holding RTL (aka ncount).
		self.write_gcr_ncount(ncount >> 16)

	def read_ncount(self):
		# mode = self.read_nsr_mode_non_ui()
		return self.registers[self.register_legend['NCOUNT'][0]]

	def read_total_ncount_non_ui(self):
		two_bytes = self.read_ncount()
		if two_bytes == 0xFFFF:
			third_byte = self.read_gcr_ncount()
			return third_byte << 16 | two_bytes
		return two_bytes

	# =========================================================================================================

	def store_firing_neuron(self, neuron):
		# Called from Neuron.update_dist(), i.e., whenever COMP or LCOMP is updated
		# Call from Neuron.broadcast()
		# Called by individual neurons whenever they fire
		log.trace("CM1KEmulator.store_firing_neuron()")

		# NOTE: firing_neurons won't be sorted until all neurons are added (see update_all_neuron_dists())

		# Only store a firing neuron if its dist-and-cat combination is unique (CM1K Hardware Manual, p. 17)
		unique = True
		for neuron2 in self.firing_neurons:
			if neuron2.dist == neuron.dist and neuron2.cat == neuron.cat:
				unique = False
				break
		if unique:
			# self.firing_neurons.append(neuron)
			insert_pos = len(self.firing_neurons)
			for i, neuron2 in enumerate(self.firing_neurons):
				if neuron2.dist <= neuron.dist:  # This must be <=, not <, so that earlier neurons win, ala CM1K spec
					insert_pos = i
					break
			self.firing_neurons.insert(insert_pos, neuron)

	def update_firing_dist_and_cat(self):
		# Called from update_all_neuron_dists() when LCOMP is updated to seed DIST with best neuron's distance
		# Called whenever DIST is read
		log.trace("CM1KEmulator.update_firing_dist_and_cat()")

		if self.firing_neurons:
			self.write_dist_non_ui(self.firing_neurons[-1].dist)
			self.write_cat(self.firing_neurons[-1].cat)
			if self.firing_neurons[-1].degenerate:
				self.write_cat_degenerate(True)
			self.firing_neurons.pop()
		else:
			self.write_dist_non_ui(0xFFFF)
			self.write_cat(0xFFFF)

	def update_all_neuron_dists(self, last_comp=False):
		# Called whenever COMP or LCOMP is updated
		log.trace("CM1KEmulator.update_all_neuron_dists()")

		gcr = self.read_gcr_context()
		comp = self.input_[self.indexcomp]
		for neuron in self.neurons:
			if (neuron.state == nrn.NeuronState.com and neuron.cxt == gcr) or neuron.state == nrn.NeuronState.rtl:
				neuron.update_dist(
					self.indexcomp, comp, self.read_ncr_norm(), last_comp, self.read_nsr_classifier_enum())
		if last_comp:
			# After writing the last component, sort the firing neurons by distance
			self.firing_neurons.sort(key=lambda x: x.dist, reverse=True)

	def broadcast(self, input_, new_gcr=None, low_level_emulation=False, aif_scale=1):
		"""
		input_ of len 1-256 (limit to 256 to properly emulate a CM1K).
		new_gcr 0-127. 1-127: context. 0: use all neurons (disregard context).
		If low_level_emulation is False, then:
			Simulate the overall behavior of a broadcast without emulating the minutiae of the chip's behavior on a
			per-component or register-level basis.
		If low_level_emulation is True, then:
			Emulate a CM1K at the low level of individual input components (bytes) and individual register updates.
			Send the input vector to the neurons one component at a time, update their distances one at a time, and read
			and write registers individually as the appropriate events occur.
		aif_scale: scalar applied to aif
		"""
		# We need to store the input in a nonlocal variable so it can be accessed from other methods on a
		# per-component (i.e., per-byte) basis, e.g., in update_all_neuron_dists().
		log.trace("CM1KEmulator.broadcast(): new_gcr: {}, input: {}".format(new_gcr, input_))

		self.input_ = input_

		if new_gcr is not None:
			self.write_gcr_context(new_gcr)

		if low_level_emulation:
			# Emulate a broadcast at a low level of abstraction, reproducing the literal chip-behavior by writing individual
			# bytes of the input pattern to the CM1K chip's registers, and updating each neuron's distance incrementally on a
			# per-byte basis.
			for i, comp in enumerate(self.input_):
				if i < len(input_) - 1:
					self.write_comp(comp)
				else:
					self.write_lcomp(comp)
		else:
			# Emulate a broadcast at a high level of abstraction by iterating over the committed neurons and updating each
			# neuron's distance from the entire input pattern just once.
			gcr = self.read_gcr_context()
			self.firing_neurons = []
			for neuron in self.neurons:
				if neuron.state == nrn.NeuronState.com and neuron.cxt == gcr:
					neuron.broadcast(self.input_, self.read_ncr_norm_enum(), self.read_nsr_classifier_enum(), aif_scale)
				elif neuron.state == nrn.NeuronState.rtl:
					# During broadcast, the input is always copied into the RTL neuron.
					# To properly emulate a CM1K, neurons should not be able to hold more than 256 bytes.
					neuron.pattern = input_
			# self.firing_neurons.sort(key=lambda x: x.dist, reverse=True)

		log.log("Num firing neurons: {}".format(len(self.firing_neurons)))
		# for neuron in self.firing_neurons:
		# 	log.log("Firing neuron: {}".format(neuron.dump()))

		self.clear_nsr_id()
		self.clear_nsr_unc()

		# NOTE: Checking if firing_neurons is empty is a mere optimization. The interior code works fine without it.
		if self.firing_neurons:
			firing_cats = set()
			for neuron in self.firing_neurons:
				firing_cats.add(neuron.cat)
			if len(firing_cats) == 1:
				self.set_nsr_id()
			elif len(firing_cats) > 1:
				self.set_nsr_unc()

		return self.read_nsr_id(), self.read_nsr_unc(), self.firing_neurons

	def learn(self, input_, cat, gcr=None):
		"""
		input_ of len 1-256 (limit to 256 to properly emulate a CM1K).
		cat 0-32767. 1-32767: vector category. 0: counterexample (shrink aifs, but don't commit RTL neuron).
		gcr 0-127. 1-127: context. 0: use all neurons (disregard context). None: don't overwrite current context.
		"""
		log.trace("CM1KEmulator.learn()")
		# log.log("Input: {},{}: {}".format(gcr, cat, input_))
		# self.dump_registers()

		self.broadcast(input_, gcr)
		self.write_cat(cat)

		min_firing_dist = self.firing_neurons[-1].dist if self.firing_neurons else self.read_maxif()
		minif = self.read_minif()

		# Shrink any misfiring neurons to the shortest distance of any firing neuron
		for neuron in self.firing_neurons:
			neuron.shrink_if_necessary(cat, min_firing_dist, minif)

		# Determine if any neurons correctly fired
		any_correctly_firing_neuron = False
		for neuron in self.firing_neurons:
			if neuron.cat == cat:
				any_correctly_firing_neuron = True
				break
		log.log("any_correctly_firing_neuron: {}".format(any_correctly_firing_neuron))

		# for neuron in self.neurons:
		# 	log.log("(A) Neuron: {}".format(neuron.dump()))

		# If there are no correctly firing neurons, consider recruiting a new neuron to hold the pattern as a new prototype
		if not any_correctly_firing_neuron:
			rtl_neuron_idx = None
			if not self.unlimited_neurons:
				# Find the RTL neuron
				# This should be stored in the registers, but it's safer to just find it explicitly
				for i, neuron in enumerate(self.neurons):
					# log.log("RTL search, neuron: {} {}".format(neuron.id_, neuron.state))
					if neuron.state == nrn.NeuronState.rtl:
						rtl_neuron_idx = i
						break
			else:
				# If the number of neurons is unlimited, then the RTL neuron is simply at the end of the list
				rtl_neuron_idx = len(self.neurons) - 1

			# Assign the RTL neuron the input as a new pattern
			if rtl_neuron_idx is not None:
				log.log("rtl_neuron: idx{}, id{}".format(rtl_neuron_idx, self.neurons[rtl_neuron_idx].id_ if self.neurons[rtl_neuron_idx] else None))
				new_aif = min(max(min_firing_dist, minif), self.read_maxif())
				log.log("new_aif: {}".format(new_aif))
				self.neurons[rtl_neuron_idx].commit(self.read_gcr_context(), cat, new_aif, input_)
				log.log("Committed: id{} st{}".format(self.neurons[rtl_neuron_idx].id_, self.neurons[rtl_neuron_idx].state))
				if self.unlimited_neurons:
					self.neurons.append(nrn.Neuron(len(self.neurons), self))
				if self.neurons[rtl_neuron_idx].id_ < len(self.neurons) - 1:
					self.neurons[self.neurons[rtl_neuron_idx + 1].id_].state = nrn.NeuronState.rtl
				self.write_total_ncount_non_ui(self.read_ncount() + 1)
			else:
				log.log("RTL neuron is None")

		# for neuron in self.neurons:
		# 	log.log("(B) Neuron: {}".format(neuron.dump()))

	def scale_all_aifs(self, scale, cxt=0):
		"""
		Scale all AIFs within the min/max AIF bounds for a given context.
		This function is obviously not part of the low-level emulator, but is in fact within the CM1K's capability.
		In order to scale the AIFs, one would have to save the network state in save-and-restore mode, edit the AIFs
		externally, and then write the updated network description back into the chip.
		My experiments failed to show any benefit to this approach. The intent was to decrease "unclassifications",
		and it worked, but only be increasing wrong classifications more than correct classifications.
		"""
		for neuron in self.neurons:
			if cxt == 0 or neuron.cxt == cxt:
				# TODO: Use the minimum and maximum AIFs of each neuron (i.e., of each context)
				neuron.aif = min(max(int(round(neuron.aif * scale)), 0), 0xFFFF)
