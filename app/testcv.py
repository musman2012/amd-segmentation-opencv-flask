import cv2 as cv
import os
import numpy as np
from PIL import Image
import io
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
from scipy import stats


def apply_median_filter(in_img):
  filtered_img = cv.medianBlur(in_img, 3)
  #show_img(filtered_img)
  return filtered_img

## Apply K-Mean Clustering (Color Quantization)
def apply_k_mean_clustering(in_img):
  Z = in_img.reshape((-1,3))
  # convert to np.float32
  Z = np.float32(Z)
  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 2
  ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
  # cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
  # Now convert back into uint8, and make original image
  center = np.uint8(center)
  res = center[label.flatten()]
  res2 = res.reshape((in_img.shape))
  return res2

def apply_canny_edge(in_img):
  edges = cv.Canny(in_img,50,200)

  return edges


def find_high_intensity_areas(in_img, th):
	resultr, resultc = [], []
	rows, cols = in_img.shape
	for col in range(cols):
		for row in range(rows):
			if in_img[row, col] > th:
				# temp = (row, col)
				resultr.append(row)
				resultc.append(col)

	return (resultr, resultc)


def find_RPE_col(rows, cols, in_loc):
	rpe_pos = []
	counter = 0

	for r in rows:
		if r == in_loc:
			break
		counter += 1

	return cols[counter]


## here we know the index of high intensity row

def detect_ilm(in_img):
	# print(edges.shape) ## 390, 508 ==> Rows x Cols
	rows, cols = in_img.shape
	center = 200  # cols//2 - 10
	center += 50
	ilm_pos = []
	ilm_dict = dict()
	seg_img = np.zeros((rows, cols), np.uint8)
	prev_row = 0
	min_row = 999
	for col in range(center, cols):
		for row in range(rows):
			if in_img[row, col] > 200:  # and (abs(prev_row - row) < 5 or prev_row == 0):
				if abs(prev_row - row) < 5 or col < 320:
					seg_img[row, col] = 255
					pos = (row, col)
					ilm_dict[col] = row
					ilm_pos.append(pos)
					min_row = min(row, min_row)
					# print("In the loop", row, col)
					prev_row = row
					break

	prev_row = 0
	for col in range(center - 1, -1, -1):
		for row in range(rows):
			if in_img[row, col] > 200:  # and (abs(prev_row - row) < 5 or prev_row == 0):
				if abs(prev_row - row) < 5 or col > 100:
					seg_img[row, col] = 255
					pos = (row, col)
					ilm_pos.append(pos)
					ilm_dict[col] = row
					min_row = min(row, min_row)
					# print("In the loop", row, col)
					prev_row = row
					break

	return ilm_pos, ilm_dict, min_row


def return_avg(starting_index, col_val, hi_cols, hi_rows):
	counter = sum = 0
	while hi_cols[starting_index] == col_val:
		sum += hi_rows[starting_index]
		starting_index += 1
		counter += 1
		if starting_index < len(hi_cols):
			break

	return sum // counter, starting_index


def segment_rpe(hi_rows, hi_cols, size):
	rpe_pos = []
	rpe_dict = dict()
	c_index = 0
	col_ptr = 0
	prev_row = 0

	for c_index in range(0, size):
		# while high_cols[c_index] == c_index:
		while col_ptr < len(hi_cols) and hi_cols[col_ptr] == c_index:
			col_ptr += 1
		temp = hi_rows[col_ptr - 1], c_index
		rpe_pos.append(temp)
		rpe_dict[c_index] = hi_rows[col_ptr - 1]
		'''
        if col_ptr < len(hi_cols) and hi_cols[col_ptr] == c_index: ## this column has certain bright spots
          aveg, ending_index = return_avg(col_ptr, c_index, hi_cols, hi_rows)
          prev_row = aveg
    
        else:
          aveg = prev_row
          temp = aveg, c_index
          rpe_pos.append(temp)
          rpe_dict[c_index] = aveg
          col_ptr += 1
        '''
	return rpe_pos, rpe_dict


def estimate_rpe(bm_pos_curve, rpe_pos_curve, total_cols):
	segmented_pos = []
	pos_dict = dict()
	max_row = 0
	for counter in range(0, 76):
		bmr, bmc = bm_pos_curve[counter]
		rper, rpec = rpe_pos_curve[counter]
		tr = min(bmr, rper)
		temp = tr, counter  # bm_pos_curve[counter]
		segmented_pos.append(temp)
		max_row = max(tr, max_row)
		pos_dict[counter] = tr

	for counter in range(76, 326):
		temp = rpe_pos_curve[counter]
		tr, tc = temp
		segmented_pos.append(temp)
		max_row = max(tr, max_row)
		pos_dict[counter] = tr

	for counter in range(326, total_cols):
		temp = bm_pos_curve[counter]
		tr, tc = temp
		segmented_pos.append(temp)
		max_row = max(tr, max_row)
		pos_dict[counter] = tr

	# counter = 350

	## starting from this counter:
	## Phase-I ==> replace rpe pos with bm_pos_curve
	## Plus do the same for first 50 cols
	##########################################
	## Phase-II ==> Check the difference in both curves and make decision

	# r,c = bm_pos_curve[counter]

	# print("Row col", r, c)

	return segmented_pos, pos_dict, max_row


def find_starting_row_fluid(ilm_dict, counter):
	if counter in ilm_dict:
		starting_row = ilm_dict[counter] + 15
		return starting_row
	else:
		for i in range(10):
			if counter - i in ilm_dict:
				starting_row = ilm_dict[counter - i]
				return starting_row

			elif counter + i in ilm_dict:
				starting_row = ilm_dict[counter + i]
				return starting_row

	return -999


def find_ending_row_fluid(rpe_dict, counter):
	if counter in rpe_dict:
		starting_row = rpe_dict[counter]
		return starting_row
	else:
		for i in range(150):
			if counter - i in rpe_dict:
				starting_row = rpe_dict[counter - i]
				return starting_row

			elif counter + i in rpe_dict:
				starting_row = rpe_dict[counter + i]
				return starting_row

	return -99


def seg_fluid(gray_img, ilm_pos, rpe_pos, thershld, thershld2, ilm_dict, rpe_dict):
	## length of rows should be equal to number of cols
	ilm_rows, ilm_cols, rpe_rows, rpe_cols = [], [], [], []
	MAX_ROW, MIN_ROW = 0, 0
	NUM_ROWS, NUM_COLS = gray_img.shape
	fluid_pos = []

	for ilms in ilm_pos:
		(row, col) = ilms
		ilm_rows.append(row)
		ilm_cols.append(col)

	for rpes in rpe_pos:
		(row, col) = rpes
		rpe_rows.append(row)
		rpe_cols.append(col)

	MAX_ROW = max(rpe_rows)
	MIN_ROW = min(ilm_rows)

	# print(len(ilm_rows))
	# print(gray_img.shape)

	counter = -1
	prev_ending_row = MAX_ROW
	prev_starting_row = MIN_ROW

	for c in range(NUM_COLS):
		counter += 1
		starting_row = find_starting_row_fluid(ilm_dict, counter)
		ending_row = find_ending_row_fluid(rpe_dict, counter)
		# print("Ending nrw", ending_row, counter, NUM_COLS)
		if ending_row == -99 or ending_row is None:
			ending_row = prev_ending_row

		if starting_row == -999:
			starting_row = prev_starting_row

		while starting_row <= ending_row:
			# for r in range(starting_row, MAX_ROW):  ## TODO-2 MAKE IT DEPENDENT ON RPE VAL HERE
			# print("Starting row is",starting_row)
			r = starting_row
			if gray_img[r, c] > thershld and gray_img[r, c] < thershld2:
				temp = (r, c)
				# resultr.append(row)
				fluid_pos.append(temp)
			starting_row += 1
		prev_ending_row = ending_row
		prev_starting_row = starting_row

	return fluid_pos, MAX_ROW, MIN_ROW


def remove_extremes(new_rows, new_cols):
	rows, cols = [], []
	r_mode = stats.mode(new_rows)
	r_mode = r_mode[0][0]

	for i in range(0, len(new_rows)):
		if r_mode - new_rows[i] > 70:
			continue
		rows.append(new_rows[i])
		cols.append(new_cols[i])

	return rows, cols


def remove_intensity_outliers(ilm_mode, ilm_dict, hi_r, hi_c):
	new_rows, new_cols = [], []
	# index = 0
	PIXEL_DIFF_TH, th2 = 50, 20
	IMP_COL = 100

	for index in range(0, len(hi_r)):
		c = hi_c[index]
		r = hi_r[index]
		flag = 0
		# print(c,r)
		if c in ilm_dict:
			if (r - ilm_dict[c]) < PIXEL_DIFF_TH:
				flag = 1
		else:
			for i in range(1, 6):
				new_c = index + i
				if new_c in ilm_dict:
					if (r - ilm_dict[new_c]) < PIXEL_DIFF_TH:
						flag = 1

				new_c = index - i
				if new_c in ilm_dict:
					if (r - ilm_dict[new_c]) < PIXEL_DIFF_TH:
						flag = 1

		if flag == 1:
			continue

		# print("Vals appended", index)
		new_rows.append(r)
		new_cols.append(c)

	#   index += 1

	new_rows, new_cols = remove_extremes(new_rows, new_cols)

	return (new_rows, new_cols)

def append_fluid_spots(fluid_spots, start_row, end_row, col, image_scan):
  while start_row < end_row and (image_scan[start_row, col] == 0 or col < 300):
    temp = start_row, col
    fluid_spots.append(temp)
    start_row += 1

  return fluid_spots

def detect_fluid_spots(detected_fluid_output, segmented_rpe_dict, ilm_dict, min_row, max_row):
	## will only be called in those scans in which the cave is clear
	## start from the ilm_dict KVP at a specific column
	## go until the value of segmented_rpe_dict[col]
	## append all the values
	tot_cols = 380  # detected_fluid_output.shape[1]
	fluid_spots = []
	s_row = min_row
	e_row = max_row
	col_start = 90
	fluid_start_row = dict()

	if col_start in ilm_dict:
		s_row = ilm_dict[col_start]

	if col_start in segmented_rpe_dict:
		e_row = segmented_rpe_dict[col_start]

	s_row += 10

	while s_row < e_row:
		if detected_fluid_output[s_row, col_start] > 0:
			fluid_spots = append_fluid_spots(fluid_spots, s_row + 1, e_row, col_start, detected_fluid_output)
			fluid_start_row[col_start] = s_row
			break
		s_row += 1

	## HERE s_row contains the location of fluid start at col 100
	## now go right and left
	prev_row = s_row
	# row = s_row - 10     ## starting from a bit up to cover any rocks going up
	for col in range(col_start + 1, tot_cols):  ## go right
		row = prev_row - 10
		if col in segmented_rpe_dict:
			e_row = segmented_rpe_dict[col]

		while row < e_row:
			if detected_fluid_output[row, col] > 100:
				fluid_spots = append_fluid_spots(fluid_spots, row + 1, e_row, col, detected_fluid_output)
				fluid_start_row[col] = row
				prev_row = row
				break
			row += 1

	prev_row = s_row
	# row = s_row - 10
	for col in range(col_start - 1, -1, -1):  ## go left
		row = prev_row - 10
		if col in segmented_rpe_dict:
			e_row = segmented_rpe_dict[col]

		while row < e_row:
			if detected_fluid_output[row, col] > 100:
				fluid_spots = append_fluid_spots(fluid_spots, row + 1, e_row, col, detected_fluid_output)
				fluid_start_row[col] = row

				prev_row = row
				break
			row += 1

	for c in range(0, tot_cols):
		if c in fluid_start_row:
			continue

		if c - 2 in fluid_start_row and c + 2 in fluid_start_row:
			e_row = segmented_rpe_dict[c]
			temp_row = (fluid_start_row[c - 2] + fluid_start_row[c + 2]) // 2
			fluid_spots = append_fluid_spots(fluid_spots, temp_row + 1, e_row, c, detected_fluid_output)
			fluid_start_row[c] = temp_row

	return fluid_spots

def return_vals(my_dict):
	vals = []
	for key in my_dict:
		vals.append(my_dict[key])

	return vals


def seg_foci(gray_img, thershld_min, thershld_max, ilm_dict, rpe_dict, MAX_ROW, MIN_ROW, RPE_ROW):
	## length of rows should be equal to number of cols

	## another algorithm
	## detect the bright clusters (Spots) first
	## pick those clusters having area of less than ___
	## only those which are below ILM and above RPE

	NUM_ROWS, NUM_COLS = gray_img.shape
	foci_pos = []

	# print("rows and cols are ", NUM_ROWS, NUM_COLS)

	counter = -1
	prev_ending_row = MAX_ROW
	prev_starting_row = MIN_ROW

	ending_row = prev_ending_row

	for c in range(NUM_COLS):
		counter += 1
		starting_row = find_starting_row_fluid(ilm_dict, counter)
		ending_row = find_ending_row_fluid(rpe_dict, counter)
		# print("Ending nrw", ending_row, counter, NUM_COLS)
		# if ending_row == -99 or ending_row is None:
		##  ending_row = prev_ending_row

		if starting_row == -999:
			starting_row = prev_starting_row

		starting_row += 20
		ending_row -= 20

		while starting_row <= ending_row and starting_row < NUM_ROWS:
			# for r in range(starting_row, MAX_ROW):  ## TODO-2 MAKE IT DEPENDENT ON RPE VAL HERE
			# print("Starting row is",starting_row)
			r = starting_row
			# if abs(r - RPE_ROW) < 10:
			#  continue

			if gray_img[r, c] > thershld_min and gray_img[r, c] < thershld_max:
				temp = (r, c)
				# resultr.append(row)
				foci_pos.append(temp)
			starting_row += 1
		prev_ending_row = ending_row
		prev_starting_row = starting_row

	return foci_pos


def cluster_fluid_edges(edges_img, segmented_rpe_dict, ilm_dict):
	empty_img = np.zeros((gray.shape[0], gray.shape[1]), np.uint8)
	t_cols = len(segmented_rpe_dict)
	# print("total cols are ", t_cols, len(segmented_rpe_dict))
	center = t_cols // 2
	lower_bound = 15
	p_start_row, p_end_row = ilm_dict[center], segmented_rpe_dict[center]
	for c in range(center - 1, -1, -1):
		start_row = p_start_row
		end_row = p_end_row
		if c in ilm_dict:
			start_row = ilm_dict[c]
		end_row = segmented_rpe_dict[c]
		empty_img[start_row + 5: end_row + lower_bound, c] = edges_img[start_row + 5: end_row + lower_bound, c]
		p_start_row = start_row
		p_end_row = end_row

	for c in range(center + 1, t_cols):
		start_row = p_start_row
		end_row = p_end_row
		if c in ilm_dict:
			start_row = ilm_dict[c]
		end_row = segmented_rpe_dict[c]
		empty_img[start_row + 5: end_row + lower_bound, c] = edges_img[start_row + 5: end_row + lower_bound, c]
		p_start_row = start_row
		p_end_row = end_row

	# for val in segmented_rpe_dict:
	# empty_img[segmented_rpe_dict[val], val] = 255

	return empty_img


def locate_fluid_from_clusters(fluid_clus, segmented_rpe_dict):
	new_fluid_pos = []
	t_rows, t_cols = fluid_clus.shape
	min_row = 100
	max_row = t_rows
	# t_cols = len(segmented_rpe_dict)
	for c in range(0, t_cols):
		max_row = segmented_rpe_dict[c]
		for r in range(min_row, t_rows):
			if fluid_clus[r, c] > 100:  ## starting row detected
				temp = (r, c)
				new_fluid_pos.append(temp)
				counter = 1
				while ((r + counter) < max_row) and fluid_clus[r + counter, c] < 100:
					temp = (r + counter, c)
					new_fluid_pos.append(temp)
					counter += 1
				break

	return new_fluid_pos


def abrupt_change_fluid(gray, foci_mn_th, foci_mx_th, ilm_dict, segmented_rpe_dict, mx_row, mn_row, rpe_row):
	t_cols = gray.shape[1]
	temp_f_pos = []
	for col in range(0, t_cols):
		min_row = mn_row
		max_row = mx_row
		if col in ilm_dict:
			min_row = ilm_dict[col]
		if col in segmented_rpe_dict:
			max_row = segmented_rpe_dict[col]

		row = min_row + 10
		prev_val = gray[row, col]
		while row < max_row:
			if abs(gray[row, col] - prev_val) > 250:
				temp = (row, col)
				temp_f_pos.append(temp)
			prev_val = gray[row, col]
			row += 5

	return temp_f_pos


### start from the ILM pos
### Go until RPE
## See if the difference is more than a specific value
## add to list

def fluid_clustering(in_img):
	Z = in_img.reshape((-1, 3))
	# convert to np.float32
	Z = np.float32(Z)
	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 3
	ret, label, center = cv.kmeans(Z, K, None, criteria, 100, cv.KMEANS_RANDOM_CENTERS)
	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((in_img.shape))
	return res2


def find_ret_contours(gray):
	rows, cols = gray.shape
	out_img = md_img.copy()
	contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	areas = []
	for cntr in contours:
		flag = 1
		area = cv.contourArea(cntr)
		areas.append(area)
		# for tpl in cntr:
		#   if 0 in tpl or rows-1 in tpl:
		#     flag = 0
		#     break
		if area < 200 or area > 200:
			# eps = 0.01 * cv.arcLength(cntr, True)
			# approx = cv.approxPolyDP(cntr, eps, True)
			out_img = cv.drawContours(out_img, [cntr], 0, (0, 255, 0), 3)  ## -1 indicates drawing ALL, then the color

	return out_img, areas

def return_dist(first_line_dict, second_line_dict):
  length = len(first_line_dict)
  distance = 0
  for i in range(100, length - 100, 10):
    if i in second_line_dict and i in first_line_dict:
      distance += abs(first_line_dict[i] - second_line_dict[i])

  return distance

def return_cave_dist(first_line_dict, second_line_dict):
  length = len(first_line_dict)
  distance = 0
  for i in range(10, 211, 10):
    if i < 50 and abs(first_line_dict[i] - first_line_dict[i-2]) > 5:
      continue
    if i in second_line_dict and i in first_line_dict:
      distance += abs(first_line_dict[i] - second_line_dict[i])

  return distance

def adjust_bm(distance, so_pos_curve_dict, so_pos_curve):
  residue = 0
  length = len(so_pos_curve_dict)
  # if distance > 300: ##
  #   residue += 10

  # if distance > 150: ##
  #   residue += 20

  # if distance > 800:
  #   residue -= 30

  for i in range(100, length - 100):
    so_pos_curve_dict[i] += residue
    temp = (so_pos_curve_dict[i], i)
    so_pos_curve[i] = temp
    #distance += abs(first_line_dict[i] - second_line_dict[i])

  return so_pos_curve_dict, so_pos_curve


def seg_ped(gray, so_pos_curve_dict, segmented_rpe_dict):
	length = len(so_pos_curve_dict)
	thresold = 130
	ped_rows = []
	ped_pos = []

	# ped_dict = dict()

	for c in range(100, length - 100):
		for r in range(segmented_rpe_dict[c], so_pos_curve_dict[c]):
			if gray[r, c] < thresold:
				temp = (r, c)
				ped_pos.append(temp)
				ped_rows.append(r)
			# ped_dict[c] =

	# temp = (so_pos_curve_dict[i], i)
	# so_pos_curve[i] = temp

	return ped_pos

def detect_roi(gray_th, ilm_pos, rpe_pos, ilm_dict, bm_pos_curve_dict):
	ilm_rows, rpe_rows = [], []
	MAX_ROW, MIN_ROW = 0, 0
	ROI = np.ones(gray_th.shape, np.uint8)
	NUM_COLS = gray_th.shape[1]

	for ilms in ilm_pos:
		(row, col) = ilms
		ilm_rows.append(row)

	for rpes in rpe_pos:
		(row, col) = rpes
		rpe_rows.append(row)

	MAX_ROW = max(rpe_rows)
	MIN_ROW = min(ilm_rows)

	for c in range(0, NUM_COLS):
		if c in ilm_dict:
			MIN_ROW = ilm_dict[c]

		if c in rpe_dict:
			MAX_ROW = rpe_dict[c]

		ROI[MIN_ROW + 5:MAX_ROW, c] = gray_th[MIN_ROW + 5:MAX_ROW, c]

	return ROI


def detect_fluid_contours(input_image, max_rpe_row, min_ilm_row):
	md_img = apply_median_filter(input_image)
	gray = cv.cvtColor(md_img, cv.COLOR_BGR2GRAY)

	chv = chan_vese(gray, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
					dt=0.5, init_level_set="checkerboard", extended_output=True)

	data = chv[1].copy()

	im_max = 255
	data = abs(data.astype(np.float64) / data.max())  # normalize the data to 0 - 1
	data = im_max * data  # Now scale by 255
	ls_img = data.astype(np.uint8)

	temp = apply_k_mean_clustering(ls_img)
	temp = apply_canny_edge(temp)
	# out_img2, areas = find_ret_contours(temp, max_rpe_row, min_ilm_row)

	return temp

def segment(scan):
	LUID_THR = 75  ## TODO1: Make this adaptive with the intensity of scan higher value more FP
	### TODO2: Separate out ROI (Region of Interest) + Add a Vertical line at left ==> Ignore open Contours
	## Not having a line at the right would help us to avoid those crocodiles
	MICRO_PER_PIXEL = 5.7

	min_thrshld, max_thrshld = 0, 65  # 65 ## TODO-1 MAKE THESE INTENSITIES DYNAMIC
	NUM_FLUID_PIXELS = NUM_FOCI_PIXELS = NUM_PED_PIXELS = 0
	RANGE_THRESHOLD = 145
	foci_mn_th, foci_mx_th = 210, 255
	RPE_THR = 200
	# img_name = image_files[9]
	# img_name = 'cropped_JB011.bmp'

	img = scan.copy()
	# img = resize_img(img)
	md_img = apply_median_filter(img)

	clus_img = apply_k_mean_clustering(md_img)
	kernel = np.ones((5, 5), np.uint8)
	erosion = cv.erode(clus_img, kernel, iterations=1)

	erosion = cv.dilate(erosion, kernel, iterations=1)

	ed_img = apply_canny_edge(erosion)
	ilm_pos, ilm_dict, min_ilm_row = detect_ilm(ed_img)

	ilm_rows, ilm_cols = [], []
	for i in range(0, len(ilm_pos)):
		if i in ilm_dict:
			ilm_cols.append(i)
			ilm_rows.append(ilm_dict[i])

	# dilated_fluid = cv.dilate(erosion, kernel,iterations = 1)
	# dilated_fluid = apply_k_mean_clustering(dilated_fluid)
	##fluid_fuzzy_clusters = fluid_clustering(md_img)

	# dbl_dil_edg = apply_canny_edge(dilated_fluid)

	gray = cv.cvtColor(md_img, cv.COLOR_BGR2GRAY)
	vals = return_vals(ilm_dict)
	ilm_mode = stats.mode(vals)
	ilm_mode = ilm_mode[0][0]
	# print("ILM MODE VALS ",ilm_mode)
	# print(np.mean(gray))
	if np.mean(gray) < 51:
		RPE_THR -= 20
		max_thrshld -= 10
	if np.mean(gray) < 43:
		RPE_THR -= 20
		max_thrshld -= 10
	# print(np.mean(gray), RPE_THR, max_thrshld)
	(hi_rows, high_cols) = find_high_intensity_areas(gray, RPE_THR)

	total_bright_points = len(hi_rows)

	(hi_rows, high_cols) = remove_intensity_outliers(ilm_mode, ilm_dict, hi_rows, high_cols)

	T_COLS = gray.shape[1]
	mode = stats.mode(hi_rows)
	rpe_row = mode[0][0]
	# print("RPE Rows and Cols are", len(hi_rows), len(high_cols))
	rpe_col = find_RPE_col(hi_rows, high_cols, rpe_row)

	rpe_pos, rpe_dict = segment_rpe(hi_rows, high_cols, gray.shape[1])

	new_hi_rows, new_hi_cols = [], []

	for pos in rpe_pos:
		r, c = pos
		new_hi_rows.append(r)
		new_hi_cols.append(c)

	rpe_pos = []
	# rpe_dict = dict()
	for i in range(0, len(high_cols)):
		temp = hi_rows[i], high_cols[i]
		rpe_pos.append(temp)

	################# Polynomial Fit for ILM ###################
	ilm_z = np.polyfit(ilm_cols, ilm_rows, 15)  ## need to use this for Foci segmentation
	ilm_equation = np.poly1d(ilm_z)

	ilm_curve = []
	ilm_curve_dict = dict()  ## pass this dict to Foci segmentation
	# rpe_dict = dict()
	for i in range(0, T_COLS):
		curve_row = int(ilm_equation(i))
		temp = curve_row, i
		ilm_curve.append(temp)
		ilm_curve_dict[i] = curve_row

	############ The idea of applying a ploynominal fit was taken from
	### Automated Segmentation of RPE Layer for the
	### Detection of Age Macular Degeneration Using OCT
	### Images

	z = np.polyfit(high_cols, hi_rows, 15)  ## need to use this for Foci segmentation
	bm = np.polyfit(high_cols, hi_rows, 3)  ## need to use this for fluid segmentation
	second = np.polyfit(high_cols, hi_rows, 2)  ## second order
	ft_or = np.polyfit(high_cols, hi_rows, 5)
	rpe_equation = np.poly1d(z)
	bm_equation = np.poly1d(bm)
	fo_equation = np.poly1d(ft_or)
	so_equation = np.poly1d(second)

	rpe_pos_curve = []
	rpe_pos_curve_dict = dict()  ## pass this dict to Foci segmentation
	# rpe_dict = dict()
	for i in range(0, T_COLS):
		curve_row = int(rpe_equation(i))
		temp = curve_row, i
		rpe_pos_curve.append(temp)
		rpe_pos_curve_dict[i] = curve_row

	bm_pos_curve = []
	bm_pos_curve_dict = dict()  ## pass this dict to Fluid segmentation
	# rpe_dict = dict()
	for i in range(0, T_COLS):
		bm_row = int(bm_equation(i))
		temp = bm_row, i
		bm_pos_curve.append(temp)
		bm_pos_curve_dict[i] = bm_row

	fo_pos_curve = []
	fo_pos_curve_dict = dict()
	# rpe_dict = dict()
	for i in range(0, T_COLS):
		fo_row = int(fo_equation(i))
		temp = fo_row, i
		fo_pos_curve.append(temp)
		fo_pos_curve_dict[i] = fo_row

	so_pos_curve = []
	so_pos_curve_dict = dict()
	# rpe_dict = dict()
	for i in range(0, T_COLS):
		so_row = int(so_equation(i))
		temp = so_row, i
		so_pos_curve.append(temp)
		so_pos_curve_dict[i] = so_row

	# print("Size of dictionaries ", len(fo_pos_curve_dict), len(bm_pos_curve_dict), len(rpe_pos_curve_dict))

	## This estimate function is taking bright points and returning average of it ==> Line Curve
	# (new_hi_rows, new_hi_cols) = estimate_rpe(new_hi_rows, new_hi_cols, rpe_col)


	# rpe_pos, rpe_dict = segment_rpe(gray, rpe_col, rpe_row, gray.shape[1])
	# print(rpe_dict)

	segmented_rpe_pos, segmented_rpe_dict, max_rpe_row = estimate_rpe(bm_pos_curve, rpe_pos_curve, T_COLS)

	## detecting Fluid
	# roi = detect_roi(md_img, ilm_pos, rpe_pos, ilm_dict, segmented_rpe_dict)
	# ret, thresh1 = cv.threshold(roi, FLUID_THR, 255, cv.THRESH_BINARY_INV)
	# gray_th = cv.cvtColor(thresh1, cv.COLOR_BGR2GRAY)
	# out_img, areas = find_ret_contours(gray_th)
	# print(areas)

	# print("Upper and lower bound ", min(ilm_pos))
	# fluid_clus = cluster_fluid_edges(dbl_dil_edg, segmented_rpe_dict, ilm_dict)  ## WRITE THIS FUNC

	########### DEBUGGING BLOCK ###########
	# fig = plt.figure(figsize=(15, 15))
	# ax = fig.add_subplot(1, 2, 1)
	# plt.imshow(img)
	# ax = fig.add_subplot(1, 2, 2)
	# plt.imshow(out_img)
	# plt.show()
	# # break
	# continue

	########### DEBUGGING BLOCK ###########

	#new_fluid_pos = locate_fluid_from_clusters(fluid_clus, segmented_rpe_dict)

	# print(rpe_dict.keys())
	# print("Length of RPE Dict is ",len(rpe_dict))
	# print(rpe_pos)

	detected_fluid_output = detect_fluid_contours(img, max_rpe_row, min_ilm_row)

	fluid_spots = detect_fluid_spots(detected_fluid_output, segmented_rpe_dict, ilm_dict, min_ilm_row, max_rpe_row)

	mid_dst = 89
	if 100 in ilm_dict:
		mid_dst = so_pos_curve_dict[100] - ilm_dict[100]

	new_dst = return_dist(ilm_dict, so_pos_curve_dict)

	if mid_dst < 100:
		max_thrshld -= 15

	#  if 100 > mid_dst >= 90:
	#    max_thrshld = 85

	fluid_pos, mx_row, mn_row = seg_fluid(gray, ilm_pos, rpe_pos, min_thrshld, max_thrshld, ilm_dict, bm_pos_curve_dict)

	foci_pos = seg_foci(gray, foci_mn_th, foci_mx_th, ilm_dict, segmented_rpe_dict, mx_row, mn_row, rpe_row)

	t_fluid = abrupt_change_fluid(gray, foci_mn_th, foci_mx_th, ilm_dict, segmented_rpe_dict, mx_row, mn_row, rpe_row)

	# count, th = 1, 10
	segmented_scan_ilm_rpe_final = img.copy()

	line_dst = return_dist(rpe_pos_curve_dict, so_pos_curve_dict)

	cave_dist = return_cave_dist(segmented_rpe_dict, ilm_curve_dict)    ## add this method

	#print("Lines distance is ", line_dst, " Brights spots", total_bright_points, "another", mid_dst)

	if cave_dist < 2000:
		final_fluid = fluid_pos.copy()
	elif cave_dist < 5000:
		final_fluid = fluid_spots.copy()

	so_pos_curve_dict, so_pos_curve = adjust_bm(line_dst, so_pos_curve_dict, so_pos_curve)

	ped_pos = seg_ped(gray, so_pos_curve_dict, segmented_rpe_dict)

	if 2100 > new_dst > 50:
		fluid_pos = []

	TWO = 0
	ZERO = 2

	for ilm_position in ilm_pos:
		(row, col) = ilm_position
		segmented_scan_ilm_rpe_final[row:row + 5, col, ZERO] = 240
		segmented_scan_ilm_rpe_final[row:row + 5, col, 1] = 10
		segmented_scan_ilm_rpe_final[row:row + 5, col, TWO] = 10

	for f in final_fluid:
		NUM_FLUID_PIXELS += 1
		(row_f, col_f) = f
		segmented_scan_ilm_rpe_final[row_f:row_f + 1, col_f, TWO] = 20
		segmented_scan_ilm_rpe_final[row_f:row_f + 1, col_f, 1] = 140
		segmented_scan_ilm_rpe_final[row_f:row_f + 1, col_f, ZERO] = 240

	for rpe_position in segmented_rpe_pos:
		(row_r, col_r) = rpe_position
		segmented_scan_ilm_rpe_final[row_r:row_r + 5, col_r, 1] = 245
		segmented_scan_ilm_rpe_final[row_r:row_r + 5, col_r, ZERO] = 20
		segmented_scan_ilm_rpe_final[row_r:row_r + 5, col_r, TWO] = 20
	# segmented_scan_ilm_rpe_final[row_r+1, col_r, 1] = 255
	# segmented_scan_ilm_rpe_final[row_r+2, col_r, 1] = 255
	# segmented_scan_ilm_rpe_final[row_r+3, col_r, 1] = 255
	# segmented_scan_ilm_rpe_final[row_r+4, col_r, 1] = 255

	for bm_position in ped_pos:
		NUM_PED_PIXELS += 1
		(row_r, col_r) = bm_position
		segmented_scan_ilm_rpe_final[row_r:row_r + 3, col_r, 1] = 255
		segmented_scan_ilm_rpe_final[row_r:row_r + 3, col_r, ZERO] = 255
		segmented_scan_ilm_rpe_final[row_r:row_r + 3, col_r, TWO] = 10

	for fc in foci_pos:
		NUM_FOCI_PIXELS += 1
		(row_r, col_r) = fc
		segmented_scan_ilm_rpe_final[row_r:row_r + 5, col_r, 1] = 64
		segmented_scan_ilm_rpe_final[row_r:row_r + 5, col_r, ZERO] = 235
		segmented_scan_ilm_rpe_final[row_r:row_r + 5, col_r, TWO] = 230

	quant_str = ""

	if NUM_FLUID_PIXELS > 10:
		quant_str += "Retinal Fluid Detected Area (um) : "
		quant_str += str(round(NUM_FLUID_PIXELS * MICRO_PER_PIXEL, 3))
		quant_str += "\n\n"


	else:
		quant_str += "No Retinal Fluid Detected in the Scan."
		quant_str += "\n\n"

	if NUM_FOCI_PIXELS > 0:
		quant_str += "Foci Detected Area (um) : "
		quant_str += str(round(NUM_FOCI_PIXELS * MICRO_PER_PIXEL, 3))
		quant_str += "\n\n"


	else:
		quant_str += "No Foci Detected in the Scan."
		quant_str += "\n\n"

	if NUM_PED_PIXELS > 0:
		quant_str += "PED Detected Area (um) : "
		quant_str += str(round(NUM_PED_PIXELS * MICRO_PER_PIXEL, 3))
		quant_str += "\n\n"


	else:
		quant_str += "No Detachment in Pigment Epithelium."
		quant_str += "\n\n"

	return segmented_scan_ilm_rpe_final, quant_str

# Load images
def image_make(file_path, f_name):
	# folder='\app\uploads'
	# path = os.path.join(folder,file_name)
	MIN_ROW = 10
	MAX_ROW = 400

	dir_path = '/templates/uploads'
	input_fname = '/app/static/input.jpg'
	output_fname = '/app/static/output.jpg'
	in_path = os.path.join(dir_path, f_name)
	out_path = os.path.join(dir_path, output_fname)
	img1 = cv.imread(file_path)

	### Crop the scan
	scan = img1[MIN_ROW:MAX_ROW,500:,:]			#crop_scan(img1)		## TODO2: Yet to write
	segmented_scan, str_to_show = segment(scan) ## TODO1: Fix Errors


	#str_to_show = "Hello"

	cv.imwrite(input_fname, img1)
	cv.imwrite(output_fname, segmented_scan)
	# img = Image.fromarray(img1.astype('uint8'))
	#
	# # create file-object in memory
	# file_object = io.BytesIO()
	#
	# # write PNG in file-object
	# img.save(file_object, 'PNG')
	#img1.save(input_fname)

	return str_to_show

#	cv2.waitKey(0)
#	cv2.destroyAllWindows()



