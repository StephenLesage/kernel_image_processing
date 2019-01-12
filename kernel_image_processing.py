from PIL import Image
import png
import multiprocessing
from multiprocessing import Pool
import numpy as np

###########################MULTIPROCESSING (YES/NO)#################################
# NO MULTIPROCESSING
multiprocess=0
# YES MULTIPROCESSING
#multiprocess=1
####################################################################################

##############################INITIAL PARAMETERS####################################
# importing initial image
#im = Image.open('initial_image_folder/squirrel.jpeg')
#im = Image.open('initial_image_folder/tiger.jpeg')
im = Image.open('initial_image_folder/zebra.jpeg')
#im = Image.open('initial_image_folder/peacock.jpeg')
#im = Image.open('initial_image_folder/lightning.jpeg')

# creating custom 3X3 kernel
#kernel = [[2., 0., 0.],[0., 1., 4],[0., 1., 0.]]

# creating custom 5X5 kernel
#kernel = [[2., 0., 0.],[0., 1., 4],[0., 1., 0.]]
####################################################################################

############################PRE-SET 3X3 KERNEL OPTIONS##############################
# IDENTITY
#kernel = [[0, 0, 0],[0, 1, 0],[0, 0, 0]]

# EDGE DETECTION KERNELS
#kernel = [[1., 0., -1.],[0., 0., 0.],[-1., 0., 1].]
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

############################PRE-SET 5X5 KERNEL OPTIONS##############################
# IDENTITY
#kernel = [[0., 0., 0., 0., 0.],[0., 0., 0., 0., 0.],[0., 0., 1., 0., 0.],[0., 0., 0., 0., 0.],[0., 0., 0., 0., 0.]]

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

# getting image dimansions and type
width,height = im.size

# split image into R G and B channels
r,g,b, = im.split()

# loading R G and B channels into numerical arrays... once in numerical arrays they can not be "merged" back into an RGB image in the traditional way because the original image has been compromised
red = r.load()
green = g.load()
blue = b.load()

# kernel loop for red image filter
def make_dat_kerrrn(kernel_color_array):
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

def five_kerns_for_me_please(kernel_color_array):
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

if multiprocess==0:
	if len(kernel) < 4:
		red_kernel_image = make_dat_kerrrn(red)
		blue_kernel_image = make_dat_kerrrn(blue)
		green_kernel_image = make_dat_kerrrn(green)

	if len(kernel) > 4:
		red_kernel_image = five_kerns_for_me_please(red)
		blue_kernel_image = five_kerns_for_me_please(blue)
		green_kernel_image = five_kerns_for_me_please(green)

if multiprocess==1:
	# because numpy is for arrays and not "images" it was weird
	# I had to reshape the images (height,width) because numoy reads column, row not row, column
	npred = np.array(r.getdata()).reshape(height, width)
	# then I had to flip all of the pixel values to across the diagonal because for some reason
	# it was like looking at the image from behind, instead of from the front. so to do this I 
	# had to rotate the matrix then flip it because there is no one movement that flips it about
	# the diagonal axis.
	npred1=np.rot90(npred)
	npred2=np.flipud(npred1)
	# print "before rotate and flip"
	# print npred[0][10]
	# print npred[10][0]
	# print red[0,10]
	# print red[10,0]
	# print "after rotate and flip"
	# print npred2[0][10]
	# print npred2[10][0]
	# print red[0,10]
	# print red[10,0]
	# print(stop)

	npgreen = np.array(g.getdata()).reshape(height, width)
	npgreen1=np.rot90(npgreen)
	npgreen2=np.flipud(npgreen1)
	npblue = np.array(b.getdata()).reshape(height, width)
	npblue1=np.rot90(npblue)
	npblue2=np.flipud(npblue1)
	npim = [npred2,npgreen2,npblue2]

	if len(kernel) < 4:
		pixelaccess = multiprocessing.Pool().map(make_dat_kerrrn, npim)

	if len(kernel) > 4:
		pixelaccess = multiprocessing.Pool().map(five_kerns_for_me_please, npim)

	red_kernel_image = pixelaccess[0]
	green_kernel_image = pixelaccess[1]
	blue_kernel_image = pixelaccess[2]



def check_the_thing(num):
	if num<0:
		return 0
	elif num>255:
		return 255
	else:
		return num

# looping through R G and B filters to create a final RGB array (final RGB image structured as [R,G,B, R,G,B, R,G,B, ...])
rgb_image =[]
x=0
y=0
z=0
print('putting filters back together...')
for i in range(height):
	temp_array=[]
	for j in range(width):
		x = check_the_thing(int(red_kernel_image[i][j]))
		y = check_the_thing(int(green_kernel_image[i][j]))
		z = check_the_thing(int(blue_kernel_image[i][j]))
		temp_array.append(x)
		temp_array.append(y)
		temp_array.append(z)
	rgb_image.append(temp_array)

# making the final RGB array into an RGB image and saving it
png.from_array(rgb_image, 'RGB').save('final_image_folder/final_image.jpg')