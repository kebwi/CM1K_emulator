"""
Keith Wiley
kwiley@keithwiley.com
http://keithwiley.com

https://archive.ics.uci.edu/ml/datasets/Mushroom
"""


def read_mushroom(data_dir):
	data = []

	print "Reading MUSHROOM dataset:"
	with open(data_dir + "agaricus-lepiota.data.txt") as f:
		for line in f.readlines():
			parts = line.strip().split(',')
			assert (len(parts) == 23)  # class, then 22 nominal features

			if parts[0] == 'e':
				cat = 0
			elif parts[0] == 'p':
				cat = 1
			else:
				raise ValueError("Unknown Mushroom data value")

			# The mushroom data consists of 22 nominal features with 126 total possible values.
			# Convert it into a 126-element boolean array (containing precisely 22 True elements) stored as bytes valued 0 or 1.
			# Thankfully, the 126 values fit within the CM1K's 256 vector length limit, leaving 130 bytes unused
			# (and only the lowest bit (of 8) used from the boolean-carrying 126 bytes).
			# Encoded this way, the maximum possible distance between two patterns will be 44, which will occur if all
			# 22 features differ between the two patterns in question. Notably, pattern-pair distances will always be even.
			features = [ord(chr(0))] * 126

			if parts[1] == 'b':
				features[0] = ord(chr(1))
			elif parts[1] == 'c':
				features[1] = ord(chr(1))
			elif parts[1] == 'x':
				features[2] = ord(chr(1))
			elif parts[1] == 'f':
				features[3] = ord(chr(1))
			elif parts[1] == 'k':
				features[4] = ord(chr(1))
			elif parts[1] == 's':
				features[5] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[2] == 'f':
				features[6] = ord(chr(1))
			elif parts[2] == 'g':
				features[7] = ord(chr(1))
			elif parts[2] == 'y':
				features[8] = ord(chr(1))
			elif parts[2] == 's':
				features[9] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[3] == 'n':
				features[10] = ord(chr(1))
			elif parts[3] == 'b':
				features[11] = ord(chr(1))
			elif parts[3] == 'c':
				features[12] = ord(chr(1))
			elif parts[3] == 'g':
				features[13] = ord(chr(1))
			elif parts[3] == 'r':
				features[14] = ord(chr(1))
			elif parts[3] == 'p':
				features[15] = ord(chr(1))
			elif parts[3] == 'u':
				features[16] = ord(chr(1))
			elif parts[3] == 'e':
				features[17] = ord(chr(1))
			elif parts[3] == 'w':
				features[18] = ord(chr(1))
			elif parts[3] == 'y':
				features[19] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[4] == 't':
				features[20] = ord(chr(1))
			elif parts[4] == 'f':
				features[21] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[5] == 'a':
				features[22] = ord(chr(1))
			elif parts[5] == 'l':
				features[23] = ord(chr(1))
			elif parts[5] == 'c':
				features[24] = ord(chr(1))
			elif parts[5] == 'y':
				features[25] = ord(chr(1))
			elif parts[5] == 'f':
				features[26] = ord(chr(1))
			elif parts[5] == 'm':
				features[27] = ord(chr(1))
			elif parts[5] == 'n':
				features[28] = ord(chr(1))
			elif parts[5] == 'p':
				features[29] = ord(chr(1))
			elif parts[5] == 's':
				features[30] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[6] == 'a':
				features[31] = ord(chr(1))
			elif parts[6] == 'd':
				features[32] = ord(chr(1))
			elif parts[6] == 'f':
				features[33] = ord(chr(1))
			elif parts[6] == 'n':
				features[34] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[7] == 'c':
				features[35] = ord(chr(1))
			elif parts[7] == 'w':
				features[36] = ord(chr(1))
			elif parts[7] == 'd':
				features[37] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[8] == 'b':
				features[38] = ord(chr(1))
			elif parts[8] == 'n':
				features[39] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[9] == 'k':
				features[40] = ord(chr(1))
			elif parts[9] == 'n':
				features[41] = ord(chr(1))
			elif parts[9] == 'b':
				features[42] = ord(chr(1))
			elif parts[9] == 'h':
				features[43] = ord(chr(1))
			elif parts[9] == 'g':
				features[44] = ord(chr(1))
			elif parts[9] == 'r':
				features[45] = ord(chr(1))
			elif parts[9] == 'o':
				features[46] = ord(chr(1))
			elif parts[9] == 'p':
				features[47] = ord(chr(1))
			elif parts[9] == 'u':
				features[48] = ord(chr(1))
			elif parts[9] == 'e':
				features[49] = ord(chr(1))
			elif parts[9] == 'w':
				features[50] = ord(chr(1))
			elif parts[9] == 'y':
				features[51] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[10] == 'e':
				features[52] = ord(chr(1))
			elif parts[10] == 't':
				features[53] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[11] == 'b':
				features[54] = ord(chr(1))
			elif parts[11] == 'c':
				features[55] = ord(chr(1))
			elif parts[11] == 'u':
				features[56] = ord(chr(1))
			elif parts[11] == 'e':
				features[57] = ord(chr(1))
			elif parts[11] == 'z':
				features[58] = ord(chr(1))
			elif parts[11] == 'r':
				features[59] = ord(chr(1))
			elif parts[11] == '?':
				features[60] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[12] == 'f':
				features[61] = ord(chr(1))
			elif parts[12] == 'y':
				features[62] = ord(chr(1))
			elif parts[12] == 'k':
				features[63] = ord(chr(1))
			elif parts[12] == 's':
				features[64] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[13] == 'f':
				features[65] = ord(chr(1))
			elif parts[13] == 'y':
				features[66] = ord(chr(1))
			elif parts[13] == 'k':
				features[67] = ord(chr(1))
			elif parts[13] == 's':
				features[68] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[14] == 'n':
				features[69] = ord(chr(1))
			elif parts[14] == 'b':
				features[70] = ord(chr(1))
			elif parts[14] == 'c':
				features[71] = ord(chr(1))
			elif parts[14] == 'g':
				features[72] = ord(chr(1))
			elif parts[14] == 'o':
				features[73] = ord(chr(1))
			elif parts[14] == 'p':
				features[74] = ord(chr(1))
			elif parts[14] == 'e':
				features[75] = ord(chr(1))
			elif parts[14] == 'w':
				features[76] = ord(chr(1))
			elif parts[14] == 'y':
				features[77] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[15] == 'n':
				features[78] = ord(chr(1))
			elif parts[15] == 'b':
				features[79] = ord(chr(1))
			elif parts[15] == 'c':
				features[80] = ord(chr(1))
			elif parts[15] == 'g':
				features[81] = ord(chr(1))
			elif parts[15] == 'o':
				features[82] = ord(chr(1))
			elif parts[15] == 'p':
				features[83] = ord(chr(1))
			elif parts[15] == 'e':
				features[84] = ord(chr(1))
			elif parts[15] == 'w':
				features[85] = ord(chr(1))
			elif parts[15] == 'y':
				features[86] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[16] == 'p':
				features[87] = ord(chr(1))
			elif parts[16] == 'u':
				features[88] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[17] == 'n':
				features[89] = ord(chr(1))
			elif parts[17] == 'o':
				features[90] = ord(chr(1))
			elif parts[17] == 'w':
				features[91] = ord(chr(1))
			elif parts[17] == 'y':
				features[92] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[18] == 'n':
				features[93] = ord(chr(1))
			elif parts[18] == 'o':
				features[94] = ord(chr(1))
			elif parts[18] == 't':
				features[95] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[19] == 'c':
				features[96] = ord(chr(1))
			elif parts[19] == 'e':
				features[97] = ord(chr(1))
			elif parts[19] == 'f':
				features[98] = ord(chr(1))
			elif parts[19] == 'l':
				features[99] = ord(chr(1))
			elif parts[19] == 'n':
				features[100] = ord(chr(1))
			elif parts[19] == 'p':
				features[101] = ord(chr(1))
			elif parts[19] == 's':
				features[102] = ord(chr(1))
			elif parts[19] == 'z':
				features[103] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[20] == 'k':
				features[104] = ord(chr(1))
			elif parts[20] == 'n':
				features[105] = ord(chr(1))
			elif parts[20] == 'b':
				features[106] = ord(chr(1))
			elif parts[20] == 'h':
				features[107] = ord(chr(1))
			elif parts[20] == 'r':
				features[108] = ord(chr(1))
			elif parts[20] == 'o':
				features[109] = ord(chr(1))
			elif parts[20] == 'u':
				features[110] = ord(chr(1))
			elif parts[20] == 'w':
				features[111] = ord(chr(1))
			elif parts[20] == 'y':
				features[112] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[21] == 'a':
				features[113] = ord(chr(1))
			elif parts[21] == 'c':
				features[114] = ord(chr(1))
			elif parts[21] == 'n':
				features[115] = ord(chr(1))
			elif parts[21] == 's':
				features[116] = ord(chr(1))
			elif parts[21] == 'v':
				features[117] = ord(chr(1))
			elif parts[21] == 'y':
				features[118] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			if parts[22] == 'g':
				features[119] = ord(chr(1))
			elif parts[22] == 'l':
				features[120] = ord(chr(1))
			elif parts[22] == 'm':
				features[121] = ord(chr(1))
			elif parts[22] == 'p':
				features[122] = ord(chr(1))
			elif parts[22] == 'u':
				features[123] = ord(chr(1))
			elif parts[22] == 'w':
				features[124] = ord(chr(1))
			elif parts[22] == 'd':
				features[125] = ord(chr(1))
			else:
				raise ValueError("Unknown Mushroom data value")

			data.append((features, cat))

	return data
