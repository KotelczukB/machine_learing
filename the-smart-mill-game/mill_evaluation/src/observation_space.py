import os
import pathlib
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

class ObservationSpace():

	def __init__(self, debug):
		self.debug = debug
		self.cur_dir = pathlib.Path(__file__).parent.absolute()
		np.set_printoptions(suppress=True)
		self.segments_of_interest = [
			1, 4, 7,
			9, 11, 13,
			17, 18, 19,
			22, 23, 24, 26, 27, 28,
			31, 32, 33,
			37, 39, 41,
			43, 46, 49
		]

	def generate_observation_space(self, img_name_prepath):
		os.chdir(self.cur_dir)

		# load model
		os.chdir('..')
		model_home = os.getcwd() + '/models/converted_keras_4/keras_model.h5'
		model = tensorflow.keras.models.load_model(model_home, compile=False)
		data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

		# generate observation space
		os.chdir('./images/output')
		return self.generate_game_state(model, data, img_name_prepath)

	def predict_image(self, model, data, path):
		size = (224, 224)
		image = Image.open(os.path.join(path))
		image = ImageOps.fit(image, size, Image.ANTIALIAS)
		image_array = np.asarray(image)
		# if self.debug:
			# image.show()
		normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
		data[0] = normalized_image_array
		prediction = model.predict(data)
		if self.debug:
			print(prediction)
		return prediction

	def generate_game_state(self, model, data, prepath):
		gs = []
		for i in self.segments_of_interest:
			path = prepath + '_' + f'{i:05}' + '.jpg'
			prediction = self.predict_image(model, data, path)
			prediction = prediction[0]
			match_element = prediction[0]
			match = 0
			for j in range(1, len(prediction)):
				if prediction[j] > match_element:
					match_element = prediction[j]
					match = j
			output = 2
			output_friendly = 'undefined'
			if match == 0:
				output = 0
				output_friendly = 'black'
			if match == 1:
				output = 1
				output_friendly = 'white'
			if match == 2:
				output = 2
				output_friendly = 'empty'
			gs.append(output)
			if self.debug:
				print(output, output_friendly)
		
		if len(self.segments_of_interest) != len(gs):
			return self.generate_game_state(model, data, prepath)
		
		return gs
