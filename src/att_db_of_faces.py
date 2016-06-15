from collections import defaultdict
import image_procs as img_p

"""
http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
"""


def read_att_db_of_faces(data_dir, invert_images=False, print_images=False):
	subdirs = [
		"s1/", "s2/", "s3/", "s4/", "s5/", "s6/", "s7/", "s8/", "s9/", "s10/",
		"s11/", "s12/", "s13/", "s14/", "s15/", "s16/", "s17/", "s18/", "s19/", "s10/",
		"s21/", "s22/", "s23/", "s24/", "s25/", "s26/", "s27/", "s28/", "s29/", "s20/",
		"s31/", "s32/", "s33/", "s34/", "s35/", "s36/", "s37/", "s38/", "s39/", "s40/",
	]

	filenames = [
		"1.pgm", "2.pgm", "3.pgm", "4.pgm", "5.pgm", "6.pgm", "7.pgm", "8.pgm", "9.pgm", "10.pgm",
	]

	labeled_images = defaultdict(list)

	# We can convert the 112x92 pixel AT&T fave images to squares by shrinking to 92x92 or growing to 112x112.
	# Both have potential advantages and disadvantages, so it is unclear which will generate superior predictive model.
	# Shrinking sacrifices potentially useful information along the top and bottom strips of the image, but requires
	# a lower subsequent resampling factor than growing to achieve the CM1K's pattern size (16x16 pixels), thus preserving
	# the central region at higher resolution in the final image. Growing preserves all pixel information in the image,
	# but when the image is later resampled to the CM1K's 256 pattern size, it will use a higher resampling factor than
	# shrinking to get the larger image down to size, thus reducing the resolution of the central region.
	crop_via_shrink = True

	# Iterate over the subdirectories (one per the forty subjects)
	for idx, subdir in enumerate(subdirs):
		label = idx + 1
		# Iterate over one subject's ten images
		for filename in filenames:
			path = data_dir + subdir + filename
			# print path

			pixels_arr_int_inv, width, height = img_p.read_pgm_file(path, invert_images)

			if crop_via_shrink:
				pixels_arr_int_inv_crp, width_crp = img_p.crop_img_to_square_via_shrink(pixels_arr_int_inv, width, height, 0)
				pixels_arr_int_inv_crp_chr = img_p.convert_image_from_num_to_chr(pixels_arr_int_inv_crp, width_crp * width_crp)
			else:
				pixels_arr_int_inv_crp, width_crp = img_p.crop_img_to_square_via_grow(pixels_arr_int_inv, width, height, 0)
				pixels_arr_int_inv_crp_chr = img_p.convert_image_from_num_to_chr(pixels_arr_int_inv_crp, width_crp * width_crp)

			if print_images:
				print "{}{}".format(subdir, filename)
				img, h = img_p.resample_image(pixels_arr_int_inv_crp, width_crp, width_crp, 16 if crop_via_shrink else 19)
				img_p.print_ascii_image(img, 16 if crop_via_shrink else 19)
				print

			labeled_images[label].append(pixels_arr_int_inv_crp_chr)

	return labeled_images
