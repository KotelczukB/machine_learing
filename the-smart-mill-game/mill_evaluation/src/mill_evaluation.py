import os
import pathlib
import shutil
import glob
from math import floor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

class MillEvaluation():

	def __init__(self, debug):
		self.debug = debug
		self.cur_dir = pathlib.Path(__file__).parent.absolute()
		self.inp_img_dir = './input'
		self.out_dir = './output'
		self.start_num = 1

	def evaluatio_mill(self, ObservationSpace, img_name):
		# input_images_list = glob.glob(self.inp_img_dir + '/*.jpg')
		os.chdir(self.cur_dir)
		os.chdir('../images')

		# split image
		self.split(self.inp_img_dir, self.start_num, img_name)

		# plot splitted images
		os.chdir(self.out_dir)
		img_name_split_wildcard = img_name[:10] + '*' + img_name[10:]
		output_images_list = sorted(glob.glob('./' + img_name_split_wildcard))
		if self.debug:
			self.image_part_plotter(output_images_list, 0)

		# generate observation space
		img_name_prepath = img_name[:10]
		return ObservationSpace.generate_observation_space(img_name_prepath)


	def dir_create(self, path):
		if (os.path.exists(path)) and (os.listdir(path) != []):
			shutil.rmtree(path)
			os.makedirs(path)
		if not os.path.exists(path):
			os.makedirs(path)
			
	def crop(self, input_file, height, width):
		img = Image.open(input_file)
		img_width, img_height = img.size
		for i in range(img_height//height):
			for j in range(img_width//width):
				box = (j*width, i*height, (j+1)*width, (i+1)*height)
				yield img.crop(box)

	def split(self, inp_img_dir, start_num, img_name):
		image_dir = self.out_dir
		self.dir_create(self.out_dir)
		self.dir_create(image_dir)
		file_num = 0

		infile_path = os.path.join(inp_img_dir, img_name)
		image = Image.open(infile_path)
		height, width = image.size
		height_split = floor(height / 7)
		width_split = floor(width / 7)
		for k, piece in enumerate(self.crop(infile_path,
										height_split, width_split), start_num):
			img = Image.new('RGB', (height_split, width_split), 255)
			img.paste(piece)
			img_path = os.path.join(image_dir, 
									img_name.split('.')[0]+ '_'
									+ str(k).zfill(5) + '.jpg')
			img.save(img_path)
		file_num += 1
		print('File ' + img_name + ' processed')

	def image_part_plotter(self, images_list, offset):
		fig = plt.figure(figsize=(14, 14))
		columns = 7
		rows = 7
		# ax enables access to manipulate each of subplots
		ax = []
		for i in range(columns*rows):
			# create subplot and append to ax
			img = mpimg.imread(images_list[i+offset])
			ax.append(fig.add_subplot(rows, columns, i+1))
			ax[-1].set_title('image ' + str(i+1))
			plt.imshow(img)
			plt.axis('off')
		plt.show() # Render the plot
