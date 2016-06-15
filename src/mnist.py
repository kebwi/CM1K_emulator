"""
Keith Wiley
kwiley@keithwiley.com
http://keithwiley.com

http://yann.lecun.com/exdb/mnist
"""

import struct
from image_procs import *


def read_mnist(data_dir, train_test, num_images_to_retrieve=None, first_image_to_retrieve=0, invert_images=False,
				crop_images_exclusion=None, center_images_exclusion=None, resampled_width=None,
			    print_images=False, write_transformed_image_files=False, verbose=0):
	"""
	Read some subset of MNIST images from disk
	"""
	if train_test != "train" and train_test != "test":
		raise ValueError("train_test must be either 'train' or 'test'")
	if crop_images_exclusion is not None and (crop_images_exclusion < 0 or crop_images_exclusion > 255):
		raise ValueError("center_images_exclusion must be (None or 0) or in the range 1-255")
	if center_images_exclusion is not None and (center_images_exclusion < 0 or center_images_exclusion > 255):
		raise ValueError("center_images_exclusion must be (None or 0) or in the range 1-255")
	if crop_images_exclusion is not None and center_images_exclusion is not None:
		raise ValueError("center_images_exclusion and crop_images_exclusion can't both be nonNone")
	if resampled_width and (resampled_width < 1):
		raise ValueError("resampled_width must be None or >=1 (the original images are 28 pixels wide)")

	if print_images:
		verbose = 1

	train_images_file = "train-images.idx3-ubyte"
	train_labels_file = "train-labels.idx1-ubyte"
	test_images_file = "t10k-images.idx3-ubyte"
	test_labels_file = "t10k-labels.idx1-ubyte"

	images_file = train_images_file if train_test == "train" else test_images_file
	labels_file = train_labels_file if train_test == "train" else test_labels_file

	labeled_images = []

	with open(data_dir + images_file, "rb") as imgs_fin:
		with open(data_dir + labels_file, "rb") as lbls_fin:
			# Read the label header

			int_bytes = lbls_fin.read(4)
			magic_num = struct.unpack(">L", int_bytes)[0]

			int_bytes = lbls_fin.read(4)
			num_images = struct.unpack(">L", int_bytes)[0]

			# Read the image header

			int_bytes = imgs_fin.read(4)
			magic_num = struct.unpack(">L", int_bytes)[0]

			int_bytes = imgs_fin.read(4)
			num_images = struct.unpack(">L", int_bytes)[0]

			int_bytes = imgs_fin.read(4)
			num_rows = struct.unpack(">L", int_bytes)[0]

			int_bytes = imgs_fin.read(4)
			num_cols = struct.unpack(">L", int_bytes)[0]

			if verbose >= 1:
				print "Magic number ", magic_num
				print "Num images: ", num_images
				print "Magic number ", magic_num
				print "Num images: ", num_images
				print "Num rows: ", num_rows
				print "Num cols: ", num_cols

			num_pixels = num_rows * num_cols

			# Seek to the first image/label to read
			imgs_fin.seek(num_rows * num_cols * first_image_to_retrieve, 1)
			lbls_fin.seek(first_image_to_retrieve, 1)

			if not num_images_to_retrieve:
				num_images_to_retrieve = num_images
			elif num_images_to_retrieve > num_images:
				num_images_to_retrieve = num_images

			# Iterate over the images
			for imgidx in xrange(0, num_images_to_retrieve):
				# print

				# Read the image's label
				byte = lbls_fin.read(1)
				label = ord(byte)
				if verbose >= 1:
					print "Label: ", label

				# Read the image's pixels, convert to array, convert to ints
				pixels = imgs_fin.read(num_pixels)
				pixels_arr = list(pixels)
				pixels_arr_int = [ord(px) for px in pixels_arr]

				if invert_images:
					# Invert the pixel values so background is high (white) and foreground ("ink") is low (black)
					pixels_arr_int_inv = []
					for i in xrange(0, num_pixels):
						px = pixels_arr_int[i]
						pixels_arr_int_inv.append(255 - px)
				else:
					pixels_arr_int_inv = pixels_arr_int

				if print_images:
					print_ascii_image(pixels_arr_int_inv, num_cols)
					print

					# # Write the original image to a PGM file
					# # Convert from a number array to a byte string
					# pixels_arr_int_inv_chr = ""
					# for i in xrange(0, num_cols * num_rows):
					# 	px = pixels_arr_int_inv[i]
					# 	pixels_arr_int_inv_chr += chr(px)
					# write_image_to_pgm_file(
					# 	mnist_dir + "mnist_{}_dim{}_img{:0>5}_lbl{}.pgm".format(train_test, num_cols, imgidx, label),
					# 	num_cols, num_rows, pixels_arr_int_inv_chr)

				# Either crop of center the image (but not both)
				crop_or_center = ""
				if crop_images_exclusion is not None and crop_images_exclusion != 0:
					crop_or_center = "CROP"
				if center_images_exclusion is not None and center_images_exclusion != 0:
					if crop_or_center != "":
						raise ValueError("Both crop and center can't be specifed. Only one, the other, or neither.")
					crop_or_center = "CENTER"

				if crop_or_center == "CROP":
					# Square crop the image
					if crop_images_exclusion is not None and crop_images_exclusion != 0:
						assert(center_images_exclusion is None)
						threshold = (256 - crop_images_exclusion) if invert_images else -(crop_images_exclusion - 1)
						pixels_arr_int_inv_cropORctr, num_rows_crp = crop_img_to_square_via_grow(
							pixels_arr_int_inv, num_cols, num_rows, threshold)
						num_cols_crp = num_rows_crp
						if print_images:
							print "Cropped image to {}".format(num_rows_crp)
							print_ascii_image(pixels_arr_int_inv_cropORctr, num_rows_crp)
					else:
						pixels_arr_int_inv_cropORctr = pixels_arr_int_inv
						num_rows_crp = num_rows
						num_cols_crp = num_rows_crp
				elif crop_or_center == "CENTER":
					# Center the image
					if center_images_exclusion is not None and center_images_exclusion != 0:
						assert(crop_images_exclusion is None)
						threshold = (256 - center_images_exclusion) if invert_images else -(center_images_exclusion - 1)
						pixels_arr_int_inv_cropORctr, hor_shift, ver_shift = center_image(
							pixels_arr_int_inv, num_cols, num_rows, threshold)
						if hor_shift or ver_shift:
							if print_images:
								print "Shifted image by {}, {}".format(hor_shift, ver_shift)
								print_ascii_image(pixels_arr_int_inv_cropORctr, num_cols)
						num_rows_crp = num_rows
						num_cols_crp = num_rows_crp
					else:
						pixels_arr_int_inv_cropORctr = pixels_arr_int_inv
						num_rows_crp = num_rows
						num_cols_crp = num_rows_crp
				else:
					pixels_arr_int_inv_cropORctr = pixels_arr_int_inv
					num_rows_crp = num_rows
					num_cols_crp = num_rows_crp

				# Resample the image (presumably to 16x16 so as to fit in the CM1K's vector size)
				if resampled_width and resampled_width != num_cols:
					pixels_arr_int_inv_cropORctr_rsz, num_rows_rsz = resample_image(
						pixels_arr_int_inv_cropORctr, num_rows_crp, num_cols_crp, resampled_width, "bilinear")
					if print_images:
						print_ascii_image(pixels_arr_int_inv_cropORctr_rsz, resampled_width)
				else:
					pixels_arr_int_inv_cropORctr_rsz = pixels_arr_int_inv_cropORctr
					num_rows_rsz = num_rows_crp
				num_pixels_rsz = len(pixels_arr_int_inv_cropORctr_rsz)

				# Convert from a number array to a byte string
				pixels_arr_int_inv_cropORctr_rsz_chr = ""
				for i in xrange(0, num_pixels_rsz):
					px = pixels_arr_int_inv_cropORctr_rsz[i]
					pixels_arr_int_inv_cropORctr_rsz_chr += chr(px)

				# Store the prepared image
				labeled_images.append((pixels_arr_int_inv_cropORctr_rsz_chr, label))

				# Write the prepared image to a PGM file
				if write_transformed_image_files and pixels_arr_int_inv_cropORctr_rsz_chr != pixels:
					write_image_to_pgm_file(
						data_dir + "mnist_{}_ctrd_dim{}_img{:0>5}_lbl{}.pgm".format(train_test, resampled_width, imgidx, label),
						resampled_width, num_rows_rsz, pixels_arr_int_inv_cropORctr_rsz_chr)

	return labeled_images
