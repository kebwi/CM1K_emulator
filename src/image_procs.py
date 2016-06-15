import math as math


def write_image_to_pgm_file(path, width, height, pixels_arr):
	"""
	Write an 8-bit image to a binary PGM file
	"""
	with open(path, 'wb') as fout:
		fout.write("P5\n")
		fout.write("{}\n".format(width))
		fout.write("{}\n".format(height))
		fout.write("255\n")
		fout.write(pixels_arr)


def read_pgm_file(path, invert_image=False):
	"""
	Read a binary PGM file.
	:return number-array, character-array, width, height
	"""
	with open(path) as fin:
		pgm_magic_number = None
		while True:
			line = fin.readline().strip()
			if line[0] == '#':
				continue
			pgm_magic_number = line
			break
		assert(pgm_magic_number == "P5")

		width = None
		height = None
		while True:
			line = fin.readline().strip()
			if line[0] == '#':
				continue
			width_height = line.split()
			assert(len(width_height) == 2)
			width = int(width_height[0])
			height = int(width_height[1])
			break
		num_pixels = width * height

		max_val = None
		while True:
			line = fin.readline().strip()
			if line[0] == '#':
				continue
			max_val = line
			break
		assert(max_val == "255")

		# Read the image's pixels, convert to array, convert to ints
		pixels = fin.read(num_pixels)
		pixels_arr = list(pixels)
		pixels_arr_int = [ord(px) for px in pixels_arr]

		if invert_image:
			# Invert the pixels so they are drawn black foreground against white background
			pixels_arr_int_inv = []
			for i in xrange(0, num_pixels):
				px = pixels_arr_int[i]
				pixels_arr_int_inv.append(255 - px)
		else:
			pixels_arr_int_inv = pixels_arr_int

		return pixels_arr_int_inv, width, height


def convert_image_from_num_to_chr(img_num, num_pixels):
	"""
	Convert from a number array to a byte string
	:return converted image
	"""
	img_chr = ""
	for i in xrange(0, num_pixels):
		px = img_num[i]
		img_chr += chr(px)
	return img_chr


def print_ascii_image(img_in, width_in):
	"""
	Draw an ASCII image.  Obviously, this requires a monospace font.
	For most fonts and line-heights, printing two characters per pixel produces a more square result.
	"""
	print "+" + ''.join(['--'] * width_in) + "+"
	row = ""
	for i, px in enumerate(img_in):
		ch = '  '

		if i % 4 == 0 and i % width_in != 0:
			ch = '| '
		if (i / width_in) % 4 == 0 and i > width_in:
			ch = '--'
		if i % 4 == 0 and i % width_in != 0 and (i / width_in) % 4 == 0 and i > width_in:
			ch = '+-'

		if px > 0:
			ch = '..'
		if px > 64:
			ch = '++'
		if px > 128:
			ch = '**'
		if px > 192:
			ch = '##'

		row += ch
		if i % width_in == width_in - 1:
			print '|' + row + '|'
			row = ""
	print "+" + ''.join(['--'] * width_in) + "+"


def resample_image(img_in, width_in, height_in, width_out, interpolation_method="bilinear"):
	"""
	Resample (i.e., interpolate) an image to new dimensions
	:return resampled image, new height
	"""
	img_out = []

	scale = float(width_out) / float(width_in)
	scale_inv = 1.0 / scale

	# print "Resampling scale and scale_inv: {}, {}".format(scale, scale_inv)

	height_out = int(height_in * scale)
	# print "Image dimensions resampled: {} R x {} C".format(height_out, width_out)

	if interpolation_method == "nearest_neighbor":
		for ro in xrange(0, height_out):
			for co in xrange(0, width_out):
				ri = int(round(float(ro) * scale_inv))
				ci = int(round(float(co) * scale_inv))
				px_nn = img_in[ri * width_in + ci]
				img_out.append(px_nn)
	elif interpolation_method == "bilinear":
		for ro in xrange(0, height_out):
			for co in xrange(0, width_out):
				ri_flt = float(ro) * scale_inv
				ri_flr = int(math.floor(ri_flt))
				ri_cln = int(math.ceil(ri_flt))
				if ri_cln == ri_flr:
					ri_cln += 1

				ci_flt = float(co) * scale_inv
				ci_flr = int(math.floor(ci_flt))
				ci_cln = int(math.ceil(ci_flt))
				if ci_cln == ci_flr:
					ci_cln += 1

				top = float(img_in[ri_flr * width_in + ci_flr]) * (ci_cln - ci_flt) \
					+ float(img_in[ri_flr * width_in + ci_cln]) * (ci_flt - ci_flr)
				bot = float(img_in[ri_cln * width_in + ci_flr]) * (ci_cln - ci_flt) \
					+ float(img_in[ri_cln * width_in + ci_cln]) * (ci_flt - ci_flr)
				center = top * (ri_cln - ri_flt) + bot * (ri_flt - ri_flr)

				px_bl = int(round(center))

				img_out.append(px_bl)
	else:
		raise ValueError("Invaliid interpolation method: ".format(interpolation_method))

	return img_out, height_out


def shift_image(img_in, width_in, height_in, hor_shift, ver_shift, bg_color):
	"""
	Shift (i.e., translate) an image without wrapping
	:return shifted image
	"""
	if hor_shift < 0:
		col_start = 0
		col_end = width_in + hor_shift
		cstep = 1
	elif hor_shift == 0:
		col_start = 0
		col_end = width_in
		cstep = 1
	else:
		col_start = width_in - 1
		col_end = hor_shift
		cstep = -1

	if ver_shift < 0:
		row_start = 0
		row_end = height_in + ver_shift
		rstep = 1
	elif ver_shift == 0:
		row_start = 0
		row_end = height_in
		rstep = 1
	else:
		row_start = height_in - 1
		row_end = ver_shift
		rstep = -1

	img_out = [bg_color] * (width_in * height_in)
	for row in xrange(row_start, row_end, rstep):
		for col in xrange(col_start, col_end, cstep):
			img_out[row * width_in + col] = img_in[(row - ver_shift) * width_in + (col - hor_shift)]
	return img_out


def find_bounding_box(img, width, height, threshold):
	"""
	Find the bounding box around foreground pixels relative to the background color.
	The background can be 0, implying high-valued foreground, or 255, implying low-valued foreground.
	The pixel value threshold used to indicate foreground pixels is configurable.
	:return bounding box left, top, right, bottom
	"""
	# Pass a positive threshold to exclude high values
	# or a nonpositive threshold to exclude low values at the inverted threshold.
	# Example: to exclude px>=255, pass 255.
	# Example: to exclude px>=250, pass 250.
	# Example: to exclude px<=5, pass -5.
	# Example: to exclude px<=0, pass 0.
	left = width - 1
	top = height - 1
	right = 0
	bottom = 0
	for row in xrange(0, height):
		for col in xrange(0, width):
			px = img[row * width + col]
			if (threshold > 0 and px < threshold) or (threshold <= 0 and px > -threshold):
				if row < top:
					top = row
				if row > bottom:
					bottom = row
				if col < left:
					left = col
				if col > right:
					right = col

	return left, top, right, bottom


def center_image(img_in, width_in, height_in, threshold):
	"""
	Center an image by bounding box (as opposed to, say, center of gravity) around the foreground pixels.
	The background can be 0, implying high-valued foreground, or 255, implying low-valued foreground.
	The pixel value threshold used to indicate foreground pixels is configurable.
	:return centered image
	"""
	left, top, right, bottom = find_bounding_box(img_in, width_in, height_in, threshold)

	right_inv = width_in - 1 - right
	bottom_inv = height_in - 1 - bottom
	hor_imbalance = right_inv - left
	ver_imbalance = bottom_inv - top
	hor_half_imbalance = hor_imbalance / 2.0
	ver_half_imbalance = ver_imbalance / 2.0
	hor_shift = int(round(hor_half_imbalance))
	ver_shift = int(round(ver_half_imbalance))

	if hor_shift == 0 and ver_shift == 0:
		return img_in, 0, 0
	return shift_image(img_in, width_in, height_in, hor_shift, ver_shift, 255 if threshold > 0 else 0), \
		hor_shift, ver_shift


def crop_img_to_square_via_grow(img_in, width_in, height_in, threshold):
	"""
	Square crop an image to a square around the foreground pixels.
	To achieve a square, growth the initially shorter dimension, filling with background color.
	:return cropped image, cropped width
	"""
	left, top, right, bottom = find_bounding_box(img_in, width_in, height_in, threshold)

	initial_cropped_width = right - left + 1
	initial_cropped_height = bottom - top + 1
	cropped_dim = initial_cropped_width

	if initial_cropped_width < initial_cropped_height:
		cropped_dim = initial_cropped_height
		diff = initial_cropped_height - initial_cropped_width
		left -= diff / 2
		right += diff / 2
		new_width = right - left + 1
		while new_width < initial_cropped_height:
			right += 1
			new_width = right - left + 1
		while new_width > initial_cropped_height:
			right -= 1
			new_width = right - left + 1
	elif initial_cropped_height < initial_cropped_width:
		cropped_dim = initial_cropped_width
		diff = initial_cropped_width - initial_cropped_height
		top -= diff / 2
		bottom += diff / 2
		new_height = bottom - top + 1
		while new_height < initial_cropped_width:
			bottom += 1
			new_height = bottom - top + 1
		while new_height > initial_cropped_width:
			bottom -= 1
			new_height = bottom - top + 1

	img_out = [255 if threshold > 0 else 0] * (cropped_dim * cropped_dim)
	for row in xrange(max(top, 0), top + cropped_dim):
		if 0 <= row < height_in:
			for col in xrange(max(left, 0), left + cropped_dim):
				if 0 <= col < width_in:
					img_out[(row - top) * cropped_dim + (col - left)] = img_in[row * width_in + col]
	return img_out, cropped_dim


def crop_img_to_square_via_shrink(img_in, width_in, height_in, threshold):
	"""
	Square crop an image to a square around the foreground pixels.
	To achieve a square, shrink the initially longer dimension.
	:return cropped image, cropped width
	"""
	left, top, right, bottom = find_bounding_box(img_in, width_in, height_in, threshold)

	initial_cropped_width = right - left + 1
	initial_cropped_height = bottom - top + 1
	cropped_dim = initial_cropped_width

	if initial_cropped_width > initial_cropped_height:
		cropped_dim = initial_cropped_height
		diff = initial_cropped_width - initial_cropped_height
		left += diff / 2
		right -= diff / 2
		new_width = right - left + 1
		while new_width > initial_cropped_height:
			right -= 1
			new_width = right - left + 1
		while new_width < initial_cropped_height:
			right += 1
			new_width = right - left + 1
	elif initial_cropped_height > initial_cropped_width:
		cropped_dim = initial_cropped_width
		diff = initial_cropped_height - initial_cropped_width
		top += diff / 2
		bottom -= diff / 2
		new_height = bottom - top + 1
		while new_height > initial_cropped_width:
			bottom -= 1
			new_height = bottom - top + 1
		while new_height < initial_cropped_width:
			bottom += 1
			new_height = bottom - top + 1

	img_out = [255 if threshold > 0 else 0] * (cropped_dim * cropped_dim)
	for row in xrange(top, top + cropped_dim):
		if row < height_in:
			for col in xrange(left, left + cropped_dim):
				if col < width_in:
					img_out[(row - top) * cropped_dim + (col - left)] = img_in[row * width_in + col]
	return img_out, cropped_dim
