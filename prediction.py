
import matplotlib.pyplot as plt
from torchvision import transforms
import tensorflow as tf
import numpy as np
import torch
import cv2
import os
def make_predictions(model, image_path,masked_path):
    #preprocess the image
	colormap = np.array([   [0,0,0],
						[128, 64, 128],
						[244, 35, 232],
						[70, 70, 70],
						[102, 102, 156],
						[190, 153, 153],
						[153, 153, 153],
						[250, 170, 30],
						[220, 220, 0],
						[107, 142, 35],
						[152, 251, 152],
						[70, 130, 180],
						[220, 20, 60],
						[255, 0, 0],
						[0, 0, 142],
						[0, 0, 70],
						[0, 60, 100],
						[0, 80, 100],
						[0, 0, 230],
						[119, 11, 32]
						])
	
	real_image=cv2.imread(image_path, cv2.IMREAD_COLOR)
	
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	image=image/255.0
	image=cv2.resize(image,(1024,512))
	image = image.astype(np.float32)
	tensor_transform=transforms.ToTensor()
	image=tensor_transform(image) #3, 800, 1200

	model.eval()
	with torch.no_grad():		
		pred = model(image.unsqueeze(0))                 # input:1,3,800,1200
		pred=torch.argmax(pred,dim=1)  #1, 800,1200
	#i = 18
	pred=pred.squeeze(0).numpy() # 800,1200


	#rgb mask
	rgb_orig_mask = cv2.imread(masked_path)[:, :, 0]
	rgb_orig_mask=cv2.resize(rgb_orig_mask,(1024, 512))

	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	ax[0].grid(False)
	ax[1].grid(False)
	ax[2].grid(False)
	ax[0].imshow(real_image)
	ax[1].imshow(colormap[pred])
	ax[2].imshow(colormap[rgb_orig_mask])
	ax[0].set_title('Real')
	ax[1].set_title('Prediction')
	ax[2].set_title('Ground truth')

	plt.show()