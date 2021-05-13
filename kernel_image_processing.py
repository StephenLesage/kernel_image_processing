from PIL import Image
import png
import multiprocessing
from multiprocessing import Pool
import numpy as np
import os

################################ INITIAL PARAMETERS ################################
#input_image = 'squirrel.jpeg'
#input_image = 'tiger.jpeg'
#input_image = 'zebra.jpeg'
#input_image = 'peacock.jpeg'
input_image = 'lightning.jpeg'
####################################################################################

########################## MULTIPROCESSING (YES/NO) ################################
# NO
multiprocess = 0
# YES
#multiprocess = 1
####################################################################################

########################### PRE-SET 3X3 KERNEL OPTIONS #############################
# IDENTITY
#kernel = [[0, 0, 0],[0, 1, 0],[0, 0, 0]]

# CUSTOM 3X3 KERNEL
#kernel = [[2., 0., 0.],[0., 1., 4],[0., 1., 0.]]

# EDGE DETECTION KERNELS
#kernel = [[1., 0., -1.],[0., 0., 0.],[-1., 0., 1.]]
#kernel = [[0., 1., 0.],[1., -4., 1.],[0., 1., 0.]]
#kernel = [[-1., -1., -1.],[-1., 8., -1.],[-1., -1., -1.]]

# SHARPEN
#kernel = [[0., -1., 0.],[-1., 5., -1.],[0., -1., 0.]]

# BOX BLUR
#kernel = [[1/9., 1/9., 1/9.],[1/9., 1/9., 1/9.],[1/9., 1/9., 1/9.]]

# GAUSSIAN BLUR
#kernel = [[1/16., 1/8., 1/16.],[1/8., 1/4., 1/8.],[1/16., 1/8., 1/16.]]
#kernel = [[1/8., 1/4., 1/8.],[1/4., 1/2., 1/4.],[1/8., 1/4., 1/8.]]
####################################################################################

########################### PRE-SET 5X5 KERNEL OPTIONS #############################
# IDENTITY
#kernel = [[0., 0., 0., 0., 0.],[0., 0., 0., 0., 0.],[0., 0., 1., 0., 0.],[0., 0., 0., 0., 0.],[0., 0., 0., 0., 0.]]

# CUSTOM 5X5 KERNEL
#kernel = [[2., 0., 0.],[0., 1., 4],[0., 1., 0.]]

# EDGE DETECTION
kernel = [[0., 0., -1., 0., 0.],[0., -1., -2., -1., 0.],[-1., -2., 16., -2., -1.],[0., -1., -2., -1., 0.],[0., 0., -1., 0., 0.]]

# BLOCK BLUR
#kernel = [[1/30.,1/30.,1/30.,1/30.,1/30.],[1/30.,1/30.,1/30.,1/30.,1/30.],[1/30.,1/30.,1/30.,1/30.,1/30.],[1/30.,1/30.,1/30.,1/30.,1/30.],[1/30.,1/30.,1/30.,1/30.,1/30.]]

# GAUSSIAN BLUR
#kernel = [[1/265., 4/265., 6/265., 4//265., 1//265.],[4/265., 16/265., 24/265., 16//265., 4//265.],[6/265., 24/265., 36/265., 24//265., 6//265.],[4/265., 16/265., 24/265., 16//265., 4//265.],[1/265., 4/265., 6/265., 4//265., 1//265.]]

# SOBEL CONVOLUTION (X-DIRECTION)
#kernel = [[2., 2., 4., 2., 2.],[1., 1., 2., 1., 1.],[0., 0., 0., 0., 0.],[-1., -1., -2., -1., -1.],[-2., -2., -4., -2., -2.]]
# SOBEL CONVOLUTION (Y-DIRECTION)
#kernel = [[2., 1., 0., -1., -2.],[2., 1., 0., -1., -2.],[4., 2., 0., -2., -4.],[2., 1., 0., -1., -2.],[2., 1., 0., -1., -2.]]

# PREWITT CONVOLUTION (X-DIRECTION)
#kernel = [[9., 9., 9., 9., 9.],[9., 5., 5., 5., 9.],[-7., -3., 0., -3., -7.],[-7., -3., -3., -3., -7.],[-7., -7., -7., -7., -7.]]
# PREWITT CONVOLUTION (Y-DIRECTION)
#kernel = [[9., 9., -7., -7., -7.],[9., 5., -3., -3., -7.],[9., 5., 0., -3., -7.],[9., 5., -3., -3., -7.],[9., 9., -7., -7., -7.]]

# PKIRSCH CONVOLUTION (X-DIRECTION)
#kernel = [[2., 2., 2., 2., 2.],[1., 1., 1., 1., 1.],[0., 0., 0., 0., 0.],[-1., -1., -1., -1., -1.],[-2., -2., -2., -2., -2.]]
# PKIRSCH CONVOLUTION (Y-DIRECTION)
#kernel = [[2., 1., 0., -1., -2.],[2., 1., 0., -1., -2.],[2., 1., 0., -1., -2.],[2., 1., 0., -1., -2.],[2., 1., 0., -1., -2.]
####################################################################################

# kernel loop for red image filter
def three_by_three( kernel_color_array ):
	print('kernel in filter...')
	kernel_image = []
	for i in range(height):
		temp_array=[]
		for j in range(width):
			c=(kernel_color_array[j,i]*kernel[1][1])
			if i!=(height-1):
				bc=(kernel_color_array[j,i+1]*kernel[2][1])
				if i==0:
					if j==0:
						rc=(kernel_color_array[j+1,i]*kernel[1][2])
						br=(kernel_color_array[j+1,i+1]*kernel[2][2])
						kernel_value=(c+rc+bc+br)
					elif j==(width-1):
						lc=(kernel_color_array[j-1,i]*kernel[1][0])
						bl=(kernel_color_array[j-1,i+1]*kernel[2][0])
						kernel_value=(c+lc+bl+bc)
					else:
						rc=(kernel_color_array[j+1,i]*kernel[1][2])
						br=(kernel_color_array[j+1,i+1]*kernel[2][2])
						lc=(kernel_color_array[j-1,i]*kernel[1][0])
						bl=(kernel_color_array[j-1,i+1]*kernel[2][0])
						kernel_value=(c+bc+rc+br+lc+bl)
				else:
					tc=(kernel_color_array[j,i-1]*kernel[0][1])
					if j==0:
						tr=(kernel_color_array[j+1,i-1]*kernel[0][2])
						rc=(kernel_color_array[j+1,i]*kernel[1][2])
						br=(kernel_color_array[j+1,i+1]*kernel[2][2])
						kernel_value=(c+tc+bc+tr+rc+br)
					elif j==(width-1):
						tl=(kernel_color_array[j-1,i-1]*kernel[0][0])
						lc=(kernel_color_array[j-1,i]*kernel[1][0])
						bl=(kernel_color_array[j-1,i+1]*kernel[2][0])
						kernel_value=(c+bc+tc+lc+tl+bl)
					else:
						tl=(kernel_color_array[j-1,i-1]*kernel[0][0])
						tr=(kernel_color_array[j+1,i-1]*kernel[0][2])
						lc=(kernel_color_array[j-1,i]*kernel[1][0])
						rc=(kernel_color_array[j+1,i]*kernel[1][2])
						bl=(kernel_color_array[j-1,i+1]*kernel[2][0])
						br=(kernel_color_array[j+1,i+1]*kernel[2][2])
						kernel_value=(tl+tc+tr+lc+c+rc+bl+bc+br)
			if i==(height-1):
				tc=(kernel_color_array[j,i-1]*kernel[0][1])
				if j==0:
					tr=(kernel_color_array[j+1,i-1]*kernel[0][2])
					rc=(kernel_color_array[j+1,i]*kernel[1][2])
					kernel_value=(c+tc+tr+rc)
				elif j==(width-1):
					tl=(kernel_color_array[j-1,i-1]*kernel[0][0])
					lc=(kernel_color_array[j-1,i]*kernel[1][0])
					kernel_value=(c+tl+tc+lc)
				else:
					tr=(kernel_color_array[j+1,i-1]*kernel[0][2])
					tl=(kernel_color_array[j-1,i-1]*kernel[0][0])
					lc=(kernel_color_array[j-1,i]*kernel[1][0])
					rc=(kernel_color_array[j+1,i]*kernel[1][2])
					kernel_value=(c+tc+tl+lc+tr+rc)
			temp_array.append(kernel_value)
		kernel_image.append(temp_array)
	return kernel_image

def five_by_five( kernel_color_array ):
	print('kernel in filter...')
	kernel_image = []
	for i in range(height):
		temp_array=[]
		for j in range(width):
			c2=(kernel_color_array[j,i]*kernel[2][2])
			if i!=(height-1) and i!=(height-2):
				d2=(kernel_color_array[j,i+1]*kernel[3][2])
				e2=(kernel_color_array[j,i+2]*kernel[4][2])
				if i==0:
					if j==0:
						c3=(kernel_color_array[j+1,i]*kernel[2][3])
						c4=(kernel_color_array[j+2,i]*kernel[2][4])
						d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
						d4=(kernel_color_array[j+2,i+1]*kernel[3][4])
						e3=(kernel_color_array[j+1,i+2]*kernel[4][3])
						e4=(kernel_color_array[j+2,i+2]*kernel[4][4])
						kernel_value=(c2+c3+c4+d2+d3+d4+e2+e3+e4)
					elif j==1:
						c1=(kernel_color_array[j-1,i]*kernel[2][1])
						c3=(kernel_color_array[j+1,i]*kernel[2][3])
						c4=(kernel_color_array[j+2,i]*kernel[2][4])
						d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
						d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
						d4=(kernel_color_array[j+2,i+1]*kernel[3][4])
						e1=(kernel_color_array[j-1,i+2]*kernel[4][1])
						e3=(kernel_color_array[j+1,i+2]*kernel[4][3])
						e4=(kernel_color_array[j+2,i+2]*kernel[4][4])
						kernel_value=(c1+c2+c3+c4+d1+d2+d3+d4+e1+e2+e3+e4)
					elif j==(width-2):
						c0=(kernel_color_array[j-2,i]*kernel[2][0])
						c1=(kernel_color_array[j-1,i]*kernel[2][1])
						d0=(kernel_color_array[j-2,+1]*kernel[3][0])
						d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
						e0=(kernel_color_array[j-2,i+2]*kernel[4][0])
						e1=(kernel_color_array[j-1,i+2]*kernel[4][1])
						c3=(kernel_color_array[j+1,i]*kernel[2][3])
						d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
						e3=(kernel_color_array[j+1,i+2]*kernel[4][3])
						kernel_value=(c0+c1+c2+c3+d0+d1+d2+d3+e0+e1+e2+e3)
					elif j==(width-1):
						c0=(kernel_color_array[j-2,i]*kernel[2][0])
						c1=(kernel_color_array[j-1,i]*kernel[2][1])
						d0=(kernel_color_array[j-2,+1]*kernel[3][0])
						d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
						e0=(kernel_color_array[j-2,i+2]*kernel[4][0])
						e1=(kernel_color_array[j-1,i+2]*kernel[4][1])
						kernel_value=(c0+c1+c2+d0+d1+d2+e0+e1+e2)
					else:
						c0=(kernel_color_array[j-2,i]*kernel[2][0])
						c1=(kernel_color_array[j-1,i]*kernel[2][1])
						c3=(kernel_color_array[j+1,i]*kernel[2][3])
						c4=(kernel_color_array[j+2,i]*kernel[2][4])
						d0=(kernel_color_array[j-2,+1]*kernel[3][0])
						d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
						d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
						d4=(kernel_color_array[j+2,i+1]*kernel[3][4])
						e0=(kernel_color_array[j-2,i+2]*kernel[4][0])
						e1=(kernel_color_array[j-1,i+2]*kernel[4][1])
						e3=(kernel_color_array[j+1,i+2]*kernel[4][3])
						e4=(kernel_color_array[j+2,i+2]*kernel[4][4])
						kernel_value=(c0+c1+c2+c3+c4+d0+d1+d2+d3+d4+e0+e1+e2+e3+e4)
				elif i==1:
					b2=(kernel_color_array[j,i-1]*kernel[1][2])
					if j==0:
						c3=(kernel_color_array[j+1,i]*kernel[2][3])
						c4=(kernel_color_array[j+2,i]*kernel[2][4])
						d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
						d4=(kernel_color_array[j+2,i+1]*kernel[3][4])
						e3=(kernel_color_array[j+1,i+2]*kernel[4][3])
						e4=(kernel_color_array[j+2,i+2]*kernel[4][4])
						b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
						b4=(kernel_color_array[j+2,i-1]*kernel[1][4])
						kernel_value=(b2+b3+b4+c2+c3+c4+d2+d3+d4+e2+e3+e4)
					elif j==1:
						c3=(kernel_color_array[j+1,i]*kernel[2][3])
						c4=(kernel_color_array[j+2,i]*kernel[2][4])
						d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
						d4=(kernel_color_array[j+2,i+1]*kernel[3][4])
						e3=(kernel_color_array[j+1,i+2]*kernel[4][3])
						e4=(kernel_color_array[j+2,i+2]*kernel[4][4])
						c1=(kernel_color_array[j-1,i]*kernel[2][1])
						d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
						e1=(kernel_color_array[j-1,i+2]*kernel[4][1])
						b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
						b4=(kernel_color_array[j+2,i-1]*kernel[1][4])
						b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
						kernel_value=(b1+b2+b3+b4+c1+c2+c3+c4+d1+d2+d3+d4+e1+e2+e3+e4)
					elif j==(width-2):
						c0=(kernel_color_array[j-2,i]*kernel[2][0])
						c1=(kernel_color_array[j-1,i]*kernel[2][1])
						d0=(kernel_color_array[j-2,+1]*kernel[3][0])
						d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
						e0=(kernel_color_array[j-2,i+2]*kernel[4][0])
						e1=(kernel_color_array[j-1,i+2]*kernel[4][1])
						c3=(kernel_color_array[j+1,i]*kernel[2][3])
						d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
						e3=(kernel_color_array[j+1,i+2]*kernel[4][3])
						b0=(kernel_color_array[j-2,i-1]*kernel[1][0])
						b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
						b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
						kernel_value=(b0+b1+b2+b3+c0+c1+c2+c3+d0+d1+d2+d3+e0+e1+e2+e3)
					elif j==(width-1):
						c0=(kernel_color_array[j-2,i]*kernel[2][0])
						c1=(kernel_color_array[j-1,i]*kernel[2][1])
						d0=(kernel_color_array[j-2,+1]*kernel[3][0])
						d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
						e0=(kernel_color_array[j-2,i+2]*kernel[4][0])
						e1=(kernel_color_array[j-1,i+2]*kernel[4][1])
						b0=(kernel_color_array[j-2,i-1]*kernel[1][0])
						b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
						kernel_value=(b0+b1+b2+c0+c1+c2+d0+d1+d2+e0+e1+e2)
					else:
						c3=(kernel_color_array[j+1,i]*kernel[2][3])
						c4=(kernel_color_array[j+2,i]*kernel[2][4])
						d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
						d4=(kernel_color_array[j+2,i+1]*kernel[3][4])
						e3=(kernel_color_array[j+1,i+2]*kernel[4][3])
						e4=(kernel_color_array[j+2,i+2]*kernel[4][4])
						b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
						b4=(kernel_color_array[j+2,i-1]*kernel[1][4])
						c0=(kernel_color_array[j-2,i]*kernel[2][0])
						c1=(kernel_color_array[j-1,i]*kernel[2][1])
						d0=(kernel_color_array[j-2,+1]*kernel[3][0])
						d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
						e0=(kernel_color_array[j-2,i+2]*kernel[4][0])
						e1=(kernel_color_array[j-1,i+2]*kernel[4][1])
						b0=(kernel_color_array[j-2,i-1]*kernel[1][0])
						b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
						kernel_value=(b0+b1+b2+b3+b4+c0+c1+c2+c3+c4+d0+d1+d2+d3+d4+e0+e1+e2+e3+e4)
				else:
					a2=(kernel_color_array[j,i-2]*kernel[0][2])
					b2=(kernel_color_array[j,i-1]*kernel[1][2])
					if j==0:
						c3=(kernel_color_array[j+1,i]*kernel[2][3])
						c4=(kernel_color_array[j+2,i]*kernel[2][4])
						d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
						d4=(kernel_color_array[j+2,i+1]*kernel[3][4])
						e3=(kernel_color_array[j+1,i+2]*kernel[4][3])
						e4=(kernel_color_array[j+2,i+2]*kernel[4][4])
						b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
						b4=(kernel_color_array[j+2,i-1]*kernel[1][4])
						a3=(kernel_color_array[j+1,i-2]*kernel[0][3])
						a4=(kernel_color_array[j+2,i-2]*kernel[0][4])
						kernel_value=(a2+a3+a4+b2+b3+b4+c2+c3+c4+d2+d3+d4+e2+e3+e4)
					elif j==1:
						a3=(kernel_color_array[j+1,i-2]*kernel[0][3])
						a4=(kernel_color_array[j+2,i-2]*kernel[0][4])
						b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
						b4=(kernel_color_array[j+2,i-1]*kernel[1][4])
						c3=(kernel_color_array[j+1,i]*kernel[2][3])
						c4=(kernel_color_array[j+2,i]*kernel[2][4])
						d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
						d4=(kernel_color_array[j+2,i+1]*kernel[3][4])
						e3=(kernel_color_array[j+1,i+2]*kernel[4][3])
						e4=(kernel_color_array[j+2,i+2]*kernel[4][4])
						c1=(kernel_color_array[j-1,i]*kernel[2][1])
						d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
						e1=(kernel_color_array[j-1,i+2]*kernel[4][1])
						b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
						a1=(kernel_color_array[j-1,i-2]*kernel[0][1])
						kernel_value=(a1+a2+a3+a4+b1+b2+b3+b4+c1+c2+c3+c4+d1+d2+d3+d4+e1+e2+e3+e4)
					elif j==(width-2):
						c0=(kernel_color_array[j-2,i]*kernel[2][0])
						c1=(kernel_color_array[j-1,i]*kernel[2][1])
						c3=(kernel_color_array[j+1,i]*kernel[2][3])
						d0=(kernel_color_array[j-2,+1]*kernel[3][0])
						d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
						d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
						e0=(kernel_color_array[j-2,i+2]*kernel[4][0])
						e1=(kernel_color_array[j-1,i+2]*kernel[4][1])
						e3=(kernel_color_array[j+1,i+2]*kernel[4][3])
						b0=(kernel_color_array[j-2,i-1]*kernel[1][0])
						b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
						b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
						a0=(kernel_color_array[j-2,i-2]*kernel[0][0])
						a1=(kernel_color_array[j-1,i-2]*kernel[0][1])
						a3=(kernel_color_array[j+1,i-2]*kernel[0][3])
						kernel_value=(a0+a1+a2+a3+b0+b1+b2+b3+c0+c1+c2+c3+d0+d1+d2+d3+e0+e1+e2+e3)
					elif j==(width-1):
						c0=(kernel_color_array[j-2,i]*kernel[2][0])
						c1=(kernel_color_array[j-1,i]*kernel[2][1])
						d0=(kernel_color_array[j-2,+1]*kernel[3][0])
						d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
						e0=(kernel_color_array[j-2,i+2]*kernel[4][0])
						e1=(kernel_color_array[j-1,i+2]*kernel[4][1])
						b0=(kernel_color_array[j-2,i-1]*kernel[1][0])
						b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
						a0=(kernel_color_array[j-2,i-2]*kernel[0][0])
						a1=(kernel_color_array[j-1,i-2]*kernel[0][1])
						kernel_value=(a0+a1+a2+b0+b1+b2+c0+c1+c2+d0+d1+d2+e0+e1+e2)
					else:
						c3=(kernel_color_array[j+1,i]*kernel[2][3])
						c4=(kernel_color_array[j+2,i]*kernel[2][4])
						d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
						d4=(kernel_color_array[j+2,i+1]*kernel[3][4])
						e3=(kernel_color_array[j+1,i+2]*kernel[4][3])
						e4=(kernel_color_array[j+2,i+2]*kernel[4][4])
						b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
						b4=(kernel_color_array[j+2,i-1]*kernel[1][4])
						a3=(kernel_color_array[j+1,i-2]*kernel[0][3])
						a4=(kernel_color_array[j+2,i-2]*kernel[0][4])
						c0=(kernel_color_array[j-2,i]*kernel[2][0])
						c1=(kernel_color_array[j-1,i]*kernel[2][1])
						d0=(kernel_color_array[j-2,+1]*kernel[3][0])
						d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
						e0=(kernel_color_array[j-2,i+2]*kernel[4][0])
						e1=(kernel_color_array[j-1,i+2]*kernel[4][1])
						b0=(kernel_color_array[j-2,i-1]*kernel[1][0])
						b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
						a0=(kernel_color_array[j-2,i-2]*kernel[0][0])
						a1=(kernel_color_array[j-1,i-2]*kernel[0][1])
						kernel_value=(a0+a1+a2+a3+a4+b0+b1+b2+b3+b4+c0+c1+c2+c3+c4+d0+d1+d2+d3+d4+e0+e1+e2+e3+e4)
			if i==(height-2):
				a2=(kernel_color_array[j,i-2]*kernel[0][2])
				b2=(kernel_color_array[j,i-1]*kernel[1][2])
				d2=(kernel_color_array[j,i+1]*kernel[3][2])
				if j==0:
					c3=(kernel_color_array[j+1,i]*kernel[2][3])
					c4=(kernel_color_array[j+2,i]*kernel[2][4])
					d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
					d4=(kernel_color_array[j+2,i+1]*kernel[3][4])
					b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
					b4=(kernel_color_array[j+2,i-1]*kernel[1][4])
					a3=(kernel_color_array[j+1,i-2]*kernel[0][3])
					a4=(kernel_color_array[j+2,i-2]*kernel[0][4])
					kernel_value=(a2+a3+a4+b2+b3+b4+c2+c3+c4+d2+d3+d4)
				elif j==1:
					c3=(kernel_color_array[j+1,i]*kernel[2][3])
					c4=(kernel_color_array[j+2,i]*kernel[2][4])
					d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
					d4=(kernel_color_array[j+2,i+1]*kernel[3][4])
					c1=(kernel_color_array[j-1,i]*kernel[2][1])
					d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
					b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
					b4=(kernel_color_array[j+2,i-1]*kernel[1][4])
					b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
					a3=(kernel_color_array[j+1,i-2]*kernel[0][3])
					a4=(kernel_color_array[j+2,i-2]*kernel[0][4])
					a1=(kernel_color_array[j-1,i-2]*kernel[0][1])
					kernel_value=(a1+a2+a3+a4+b1+b2+b3+b4+c1+c2+c3+c4+d1+d2+d3+d4)
				elif j==(width-2):
					c0=(kernel_color_array[j-2,i]*kernel[2][0])
					c1=(kernel_color_array[j-1,i]*kernel[2][1])
					d0=(kernel_color_array[j-2,+1]*kernel[3][0])
					d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
					c3=(kernel_color_array[j+1,i]*kernel[2][3])
					d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
					b0=(kernel_color_array[j-2,i-1]*kernel[1][0])
					b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
					b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
					a0=(kernel_color_array[j-2,i-2]*kernel[0][0])
					a1=(kernel_color_array[j-1,i-2]*kernel[0][1])
					a3=(kernel_color_array[j+1,i-2]*kernel[0][3])
					kernel_value=(a0+a1+a2+a3+b0+b1+b2+b3+c0+c1+c2+c3+d0+d1+d2+d3)
				elif j==(width-1):
					c0=(kernel_color_array[j-2,i]*kernel[2][0])
					c1=(kernel_color_array[j-1,i]*kernel[2][1])
					d0=(kernel_color_array[j-2,+1]*kernel[3][0])
					d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
					b0=(kernel_color_array[j-2,i-1]*kernel[1][0])
					b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
					a0=(kernel_color_array[j-2,i-2]*kernel[0][0])
					a1=(kernel_color_array[j-1,i-2]*kernel[0][1])
					kernel_value=(a0+a1+a2+b0+b1+b2+c0+c1+c2+d0+d1+d2)
				else:
					c3=(kernel_color_array[j+1,i]*kernel[2][3])
					c4=(kernel_color_array[j+2,i]*kernel[2][4])
					d3=(kernel_color_array[j+1,i+1]*kernel[3][3])
					d4=(kernel_color_array[j+2,i+1]*kernel[3][4])
					b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
					b4=(kernel_color_array[j+2,i-1]*kernel[1][4])
					a3=(kernel_color_array[j+1,i-2]*kernel[0][3])
					a4=(kernel_color_array[j+2,i-2]*kernel[0][4])
					c0=(kernel_color_array[j-2,i]*kernel[2][0])
					c1=(kernel_color_array[j-1,i]*kernel[2][1])
					d0=(kernel_color_array[j-2,+1]*kernel[3][0])
					d1=(kernel_color_array[j-1,i+1]*kernel[3][1])
					b0=(kernel_color_array[j-2,i-1]*kernel[1][0])
					b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
					a0=(kernel_color_array[j-2,i-2]*kernel[0][0])
					a1=(kernel_color_array[j-1,i-2]*kernel[0][1])
					kernel_value=(a0+a1+a2+a3+a4+b0+b1+b2+b3+b4+c0+c1+c2+c3+c4+d0+d1+d2+d3+d4)
			if i==(height-1):
				a2=(kernel_color_array[j,i-2]*kernel[0][2])
				b2=(kernel_color_array[j,i-1]*kernel[1][2])
				if j==0:
					c3=(kernel_color_array[j+1,i]*kernel[2][3])
					c4=(kernel_color_array[j+2,i]*kernel[2][4])
					b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
					b4=(kernel_color_array[j+2,i-1]*kernel[1][4])
					a3=(kernel_color_array[j+1,i-2]*kernel[0][3])
					a4=(kernel_color_array[j+2,i-2]*kernel[0][4])
					kernel_value=(a2+a3+a4+b2+b3+b4+c2+c3+c4)
				elif j==1:
					c3=(kernel_color_array[j+1,i]*kernel[2][3])
					c4=(kernel_color_array[j+2,i]*kernel[2][4])
					c1=(kernel_color_array[j-1,i]*kernel[2][1])
					b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
					b4=(kernel_color_array[j+2,i-1]*kernel[1][4])
					b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
					a3=(kernel_color_array[j+1,i-2]*kernel[0][3])
					a4=(kernel_color_array[j+2,i-2]*kernel[0][4])
					a1=(kernel_color_array[j-1,i-2]*kernel[0][1])
					kernel_value=(a1+a2+a3+a4+b1+b2+b3+b4+c1+c2+c3+c4)
				elif j==(width-2):
					c0=(kernel_color_array[j-2,i]*kernel[2][0])
					c1=(kernel_color_array[j-1,i]*kernel[2][1])
					c3=(kernel_color_array[j+1,i]*kernel[2][3])
					b0=(kernel_color_array[j-2,i-1]*kernel[1][0])
					b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
					b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
					a0=(kernel_color_array[j-2,i-2]*kernel[0][0])
					a1=(kernel_color_array[j-1,i-2]*kernel[0][1])
					a3=(kernel_color_array[j+1,i-2]*kernel[0][3])
					kernel_value=(a0+a1+a2+a3+b0+b1+b2+b3+c0+c1+c2+c3)
				elif j==(width-1):
					c0=(kernel_color_array[j-2,i]*kernel[2][0])
					c1=(kernel_color_array[j-1,i]*kernel[2][1])
					b0=(kernel_color_array[j-2,i-1]*kernel[1][0])
					b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
					a0=(kernel_color_array[j-2,i-2]*kernel[0][0])
					a1=(kernel_color_array[j-1,i-2]*kernel[0][1])
					kernel_value=(a0+a1+a2+b0+b1+b2+c0+c1+c2)
				else:
					c3=(kernel_color_array[j+1,i]*kernel[2][3])
					c4=(kernel_color_array[j+2,i]*kernel[2][4])
					b3=(kernel_color_array[j+1,i-1]*kernel[1][3])
					b4=(kernel_color_array[j+2,i-1]*kernel[1][4])
					a3=(kernel_color_array[j+1,i-2]*kernel[0][3])
					a4=(kernel_color_array[j+2,i-2]*kernel[0][4])
					c0=(kernel_color_array[j-2,i]*kernel[2][0])
					c1=(kernel_color_array[j-1,i]*kernel[2][1])
					b0=(kernel_color_array[j-2,i-1]*kernel[1][0])
					b1=(kernel_color_array[j-1,i-1]*kernel[1][1])
					a0=(kernel_color_array[j-2,i-2]*kernel[0][0])
					a1=(kernel_color_array[j-1,i-2]*kernel[0][1])
					kernel_value=(a0+a1+a2+a3+a4+b0+b1+b2+b3+b4+c0+c1+c2+c3+c4)
			temp_array.append(kernel_value)
		kernel_image.append(temp_array)
	return kernel_image

def correct_for_overflow_values( value ):
	if value < 0:
		return 0
	elif value > 255:
		return 255
	else:
		return value

# check if file output directory exists, if not make it
if not os.path.exists( os.getcwd()+'/final_image_folder' ):
    os.mkdir( os.getcwd()+'/final_image_folder' )

# setup file input and output directory
initial_image_folder = 'initial_image_folder'
final_image_folder = 'final_image_folder'

# import initial image
im = Image.open(initial_image_folder+'/'+input_image)

# get image dimensions and type
width,height = im.size

# split image into R, G, and B channels
r,g,b, = im.split()

# loading R, G, and B channels into numerical arrays... once in numerical arrays they can not be "merged" back into an RGB image in the traditional way because the original image has been compromised
red = r.load()
green = g.load()
blue = b.load()

# no multiprocessing
if multiprocess == 0:
	if len( kernel ) < 4:
		red_kernel_image = three_by_three(red)
		blue_kernel_image = three_by_three(blue)
		green_kernel_image = three_by_three(green)
	if len( kernel ) > 4:
		red_kernel_image = five_by_five(red)
		blue_kernel_image = five_by_five(blue)
		green_kernel_image = five_by_five(green)

# yes multiprocessing
if multiprocess == 1:
	# first we reshape the images (height,width) because numpy reads column,row not row,column
	# then, for some reason it was looking at the image from behind, so we have to flip the pixels across the diagonal
	# but there is no function to do this, so first we rotate the matrix then we flip it about the diagonal
	npred = np.array(r.getdata()).reshape(height, width)
	npred1 = np.rot90(npred)
	npred2 = np.flipud(npred1)
	npgreen = np.array(g.getdata()).reshape(height, width)
	npgreen1 = np.rot90(npgreen)
	npgreen2 = np.flipud(npgreen1)
	npblue = np.array(b.getdata()).reshape(height, width)
	npblue1 = np.rot90(npblue)
	npblue2 = np.flipud(npblue1)
	npim = [npred2,npgreen2,npblue2]

	if len(kernel) < 4:
		pixelaccess = multiprocessing.Pool().map(three_by_three, npim)
	if len(kernel) > 4:
		pixelaccess = multiprocessing.Pool().map(five_by_five, npim)

	red_kernel_image = pixelaccess[0]
	green_kernel_image = pixelaccess[1]
	blue_kernel_image = pixelaccess[2]

# looping through R, G, and B filters to create a final RGB array (final RGB image structured as [R,G,B, R,G,B, R,G,B, ...])
rgb_image = []
x = 0
y = 0
z = 0
print('putting filters back together...')
for i in range( height ):
	temp_array = []
	for j in range( width ):
		x = correct_for_overflow_values(int(red_kernel_image[i][j]))
		y = correct_for_overflow_values(int(green_kernel_image[i][j]))
		z = correct_for_overflow_values(int(blue_kernel_image[i][j]))
		temp_array.append(x)
		temp_array.append(y)
		temp_array.append(z)
	rgb_image.append(temp_array)

# transforming the final RGB array into an RGB image and saving it
png.from_array(rgb_image, 'RGB').save(final_image_folder+'/'+input_image.split('.')[0]+'_final.jpg')



