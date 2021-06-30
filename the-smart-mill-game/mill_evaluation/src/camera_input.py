import os
import pathlib
import cv2

class CameraInput():

	def __init__(self):
		self.cur_dir = pathlib.Path(__file__).parent.absolute()
		self.camera = cv2.VideoCapture(0)
		self.cam_frame_name = 'The Smart Mill Game'

	def open_camera(self, MillEvaluation, ObservationSpace):
		cv2.namedWindow(self.cam_frame_name)
		while True:
			ret, frame = self.camera.read()
			if not ret:
				print('failed to grab frame')
				break

			height, width, channels = frame.shape
			indent = 50
			upper_left = (((width - height) // 2) + indent, indent)
			bottom_right = (width - (((width - height) // 2) + indent), height - indent)

			# draw in the image
			cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 0), thickness = 1)
			crop_frame = frame[upper_left[1] + 1 : bottom_right[1] - 1, upper_left[0] + 1: bottom_right[0] - 1]
			cv2.imshow(self.cam_frame_name, frame)

			k = cv2.waitKey(1)
			if k%256 == 27:
				# ESC pressed
				print('Escape hit, closing...')
				break
			elif k%256 == 32:
				# SPACE pressed
				img_name = self.store_img(crop_frame)
				return MillEvaluation.evaluatio_mill(ObservationSpace, img_name)

		self.camera.release()
		cv2.destroyAllWindows()


	def store_img(self, crop_frame):
		i = 1
		os.chdir(self.cur_dir)
		os.chdir('../images/input')
		while os.path.exists('mill_' + f'{i:05}' + '.jpg'):
			i += 1
		img_name = 'mill_' + f'{i:05}' + '.jpg'
		cv2.imwrite(img_name, crop_frame)
		print('File ' + img_name + ' written')
		return img_name
