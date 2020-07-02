import base64
import numpy as np
import io
import PIL
from PIL import Image#
# import keras
import cv2
# import matplotlib.pyplot as plt
import os
# from keras import backend as K
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras as k
from keras.models import Sequential
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask import render_template
from scipy import signal
import cv2
import sys
import glob
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import os
from PIL import Image
import xlrd 
import math
from pylab import*
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data

app = Flask(__name__,template_folder='static')
#blood vessels
def extract_bvl(image):
    b,green_fundus,r = cv2.split(image)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)
  # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)

  # removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)			
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	

  # removing blobs of microaneurysm & unwanted bigger chunks taking in consideration they are not straight lines like blood
  # vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"	
        else:
            shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)
    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
    blood_vessels = cv2.bitwise_not(finimage)
    dilated = cv2.erode(blood_vessels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=1)
  #dilated1 = cv2.dilate(blood_vessels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
    blood_vessels_1 = cv2.bitwise_not(dilated)
    return blood_vessels_1
def threshl(mic, ex, blod)
    
def dispbv(image):

    bloodvessel = extract_bvl(image)
    n_white_pix = np.sum(bloodvessel == 255)
    return n_white_pix

#microaneurysm
def dispma(image):
        # image = cv2.resize(np.float32(image), (4800, 4800))
        bloodvessel = extract_ma(image)
#         imag = rgb2gray(bloodvessel)
        # plt.imshow(bloodvessel,cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        n_white_pix = np.sum(bloodvessel == 255)
        print(n_white_pix)
        # print('Number of white pixels:', n_white_pix)
        return n_white_pix
def rgb2gray(rgb):
    
    r, g, b = cv2.split(np.float32(rgb))
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray 
def adjust_gamma(image, gamma=1.0):

   
   table = np.array([((i / 255.0) ** gamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   print(type(table))   
   print(table.ndim)
   print(table.shape)
   print(table.size)
   print(table.dtype)
   print(image.dtype)
   print(type(image))   
   print(image.ndim)
   print(image.shape)
   print(image.size)
   return cv2.LUT(image, table)
   
def extract_ma(image):
    # image = preprocess_image(image)
    # grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    r,g,b=cv2.split(image)
    comp=255-g
    # im = rgb2gray(image)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    
    histe=clahe.apply(comp)
    adjustImage = adjust_gamma(histe,gamma=3)
    comp = 255-adjustImage
    J =  adjust_gamma(comp,gamma=4)
    J = 255-J
    J = adjust_gamma(J,gamma=4)
   
    K=np.ones((11,11),np.float32)
    L = cv2.filter2D(J,-1,K)
   
    ret3,thresh2 = cv2.threshold(L,125,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    kernel2=np.ones((9,9),np.uint8)
    tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
    kernel3=np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)
    return opening

## exudates
def disp(image):
        # image = cv2.resize(np.float32(image), (4800, 4800))
        bloodvessel = extract_ex(image)
#         imag = rgb2gray(bloodvessel)
        # plt.imshow(bloodvessel,cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        n_white_pix = np.sum(bloodvessel == 255)
        print(n_white_pix)
        # print('Number of white pixels:', n_white_pix)
        return n_white_pix
def get_average_intensity(green_channel):
	average_intensity = green_channel.copy()
	i = 0
	j = 0
	while i < green_channel.shape[0]:
		j = 0
		while j < green_channel.shape[1]:
			sub_image = green_channel[i:i+20,j:j+25]
			mean = np.mean(sub_image)
			average_intensity[i:i+20,j:j+25] = mean
			j = j+25
		i = i+20
	result = np.reshape(average_intensity, (average_intensity.size,1))
	return result
def get_average_hue(hue_image):
	average_hue = hue_image.copy()
	i = 0
	j = 0
	while i < hue_image.shape[0]:
		j = 0
		while j < hue_image.shape[1]:
			sub_image = hue_image[i:i+20,j:j+25]
			mean = np.mean(sub_image)
			average_hue[i:i+20,j:j+25] = mean
			j = j+25
		i = i+20
	result = np.reshape(average_hue, (average_hue.size,1))
	return result
def get_average_saturation(hue_image):
	average_hue = hue_image.copy()
	i = 0
	j = 0
	while i < hue_image.shape[0]:
		j = 0
		while j < hue_image.shape[1]:
			sub_image = hue_image[i:i+20,j:j+25]
			mean = np.mean(sub_image)
			average_hue[i:i+20,j:j+25] = mean
			j = j+25
		i = i+20
	result = np.reshape(average_hue, (average_hue.size,1))
	return result
def extract_bv(image):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	contrast_enhanced_green_fundus = clahe.apply(image)
	# applying alternate sequential filtering (3 times closing opening)
	r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
	R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
	f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
	f5 = clahe.apply(f4)

	# removing very small contours through area parameter noise removal
	ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
	mask = np.ones(f5.shape[:2], dtype="uint8") * 255
	contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt) <= 200:
			cv2.drawContours(mask, [cnt], -1, 0, -1)			
	im = cv2.bitwise_and(f5, f5, mask=mask)
	ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
	newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	

	# removing blobs of microaneurysm & unwanted bigger chunks taking in consideration they are not straight lines like blood
	# vessels and also in an interval of area
	fundus_eroded = cv2.bitwise_not(newfin)
	xmask = np.ones(image.shape[:2], dtype="uint8") * 255
	xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
	for cnt in xcontours:
		shape = "unidentified"
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
		if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
			shape = "circle"	
		else:
			shape = "veins"
		if(shape=="circle"):
			cv2.drawContours(xmask, [cnt], -1, 0, -1)	
	
	finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
	blood_vessels = cv2.bitwise_not(finimage)
	dilated = cv2.erode(blood_vessels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=1)
	dilated1 = cv2.dilate(blood_vessels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
	blood_vessels_1 = cv2.bitwise_not(dilated)
	return blood_vessels_1
def get_SD_data(sd_image):	
	feature_1 = np.reshape(sd_image, (sd_image.size,1))
	return feature_1
def get_HUE_data(hue_image):	
	feature_2 = np.reshape(hue_image,(hue_image.size,1))	
	return feature_2
def get_saturation_data(s_image):
	feature = np.reshape(s_image,(s_image.size,1))	
	return feature
def get_INTENSITY_data(intensity_image):	
	feature_3 = np.reshape(intensity_image,(intensity_image.size,1))	
	return feature_3
def get_EDGE_data(edge_candidates_image):
	feature_4 = np.reshape(edge_candidates_image,(edge_candidates_image.size,1))	
	return feature_4
def get_RED_data(red_channel):	
	feature_5 = np.reshape(red_channel, (red_channel.size,1))	
	return feature_5
def get_GREEN_data(green_channel):
	feature_6 = np.reshape(green_channel, (green_channel.size,1))	
	return feature_6
def edge_pixel_image(image,bv_image):
	edge_result = image.copy()
	edge_result = cv2.Canny(edge_result,30,100)	
	i = 0
	j = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			if edge_result[i,j] == 255 and bv_image[i,j] == 255:
				edge_result[i,j] = 0
			j = j+1
		i = i+1
	newfin = cv2.dilate(edge_result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
	return newfin
def deviation_from_mean(image):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_output = clahe.apply(image)
	print(clahe_output)
	result = clahe_output.copy()
	result = result.astype('int')
	i = 0
	j = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			sub_image = clahe_output[i:i+5,j:j+5]
			mean = np.mean(sub_image)
			sub_image = sub_image - mean
			result[i:i+5,j:j+5] = sub_image
			j = j+5
		i = i+5
	return result   
def standard_deviation_image(image):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_output = clahe.apply(image)
	result = clahe_output.copy()
	i = 0
	j = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			sub_image = clahe_output[i:i+20,j:j+25]
			var = np.var(sub_image)
			result[i:i+20,j:j+25] = var
			j = j+25
		i = i+20
	return result 

def optic(image):
    Abo,Ago,Aro = cv2.split(image)
	
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    Ago = clahe.apply(Ago)
    M = 60    #filter size
    filter = signal.gaussian(M, std=6) #Gaussian Window
    filter=filter/sum(filter)
    STDf = filter.std()  #It'standard deviation
    Ar = Aro - Aro.mean() - Aro.std() #Preprocessing Red
    Mr = Ar.mean()                           #Mean of preprocessed red
    SDr = Ar.std()                           #SD of preprocessed red
    Thr = 1.8*M - STDf - Ar.std() 
    print(Thr) 
    Ag = Ago - Ago.mean() - Ago.std()   
    r,c = Ag.shape
    Dd = np.zeros(shape=(r,c)) #Segmented disc image initialization
    Dc = np.zeros(shape=(r,c)) #Segmented cup image initialization

    #Using obtained threshold for thresholding of the fundus image
    for i in range(1,r):
        for j in range(1,c):
            if Ar[i,j]>Thr:
                image[i,j]= [0,0,0]
            else:
                image[i,j]=image[i,j]
    return image

def extract_ex(img):
  fundus = cv2.resize(np.uint8(img),(800,615))
  print(fundus)
  fundus=optic(fundus)	
  print(fundus)	
  fundus_mask = cv2.imread(os.path.expanduser( "/home/ec2-user/environment/environment/wtproj/flaserve/static/fmask.png"))

  print(fundus_mask)
  fundus_mask = cv2.resize(np.uint8(fundus_mask),(800,615))
  f1 = cv2.bitwise_and(fundus[:,:,0],fundus_mask[:,:,0])
  f2 = cv2.bitwise_and(fundus[:,:,1],fundus_mask[:,:,1])
  f3 = cv2.bitwise_and(fundus[:,:,2],fundus_mask[:,:,2])
  fundus_dash = cv2.merge((f1,f2,f3))
  b,g,r = cv2.split(fundus_dash)		
  hsv_fundus = cv2.cvtColor(fundus_dash,cv2.COLOR_BGR2HSV)
  h,s,v = cv2.split(hsv_fundus)		
  gray_scale = cv2.cvtColor(fundus_dash,cv2.COLOR_BGR2GRAY)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  contrast_enhanced_fundus = clahe.apply(gray_scale)		
  contrast_enhanced_green_fundus = clahe.apply(g)
  plt.imshow(contrast_enhanced_green_fundus)
  average_intensity = get_average_intensity(contrast_enhanced_green_fundus)/255
  average_hue = get_average_hue(h)/255
  average_saturation = get_average_saturation(s)/255	
  bv_image_dash = extract_bv(g)
  bv_image = extract_bv(gray_scale)
  var_fundus = standard_deviation_image(contrast_enhanced_fundus)
  edge_feature_output = edge_pixel_image(gray_scale,bv_image)		
  newfin = cv2.dilate(edge_feature_output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)
  edge_candidates = cv2.erode(newfin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)		
  edge_candidates = np.uint8(edge_candidates)	
  plt.imshow(edge_candidates)
                                
      # label_image = cv2.imread(LabelFolder+'/'+file_name_no_extension+"_final_label.bmp")
  deviation_matrix = deviation_from_mean(gray_scale)

  feature1 = get_SD_data(var_fundus)/255
  feature2 = get_HUE_data(h)/255
  feature3 = get_saturation_data(s)/255
  feature4 = get_INTENSITY_data(contrast_enhanced_fundus)/255
  feature5 = get_RED_data(r)/255
  feature6 = get_GREEN_data(g)/255
  feature8 = get_HUE_data(deviation_matrix)/255
  print(feature8.shape,"deviation data shape")


  Z = np.hstack((feature2,feature3))	#HUE and SATURATION
  Z = np.float32(Z)

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
  ret,label,center=cv2.kmeans(Z,6,None,criteria,50,cv2.KMEANS_RANDOM_CENTERS)	

  u, indices, counts = np.unique(label, return_index=True, return_counts=True)	
      
  center_t = [(t[0]*255,t[1]*255) for t in center]		
  ex_color = (40,230)

  distance = [(abs(t[0]- ex_color[0]),t) for t in center_t]								
  index1 = distance.index((min(distance)))
  if counts[distance.index((min(distance)))] > 0.2*gray_scale.shape[0]*gray_scale.shape[1]:
      index1 = -1		

  distance2 = [(abs(t[0]- ex_color[0])+abs(t[1]-ex_color[1]),t) for t in center_t]		
  index2 = -1		
  if min(distance2)[0] <=25:
      index2 = distance2.index((min(distance2)))		
  if counts[distance2.index((min(distance2)))] > 0.2*gray_scale.shape[0]*gray_scale.shape[1]:
      index2 = -1

  green = [0,255,0]
  blue = [255,0,0]
  red = [0,0,255]
  white = [255,255,255]
  black = [0,0,0]
  pink = [220,30,210]
  sky = [30,240,230]
  yellow = [230,230,30]

  color = [white,black,red,green,blue,pink]
  color = np.array(color,np.uint8)
  label = np.reshape(label, gray_scale.shape)

  plt.imshow(label)


  test = label.copy()		
  if index1 == -1:
      test.fill(0)
  else:
      test[test!=distance.index((min(distance)))] = -1
      test[test==distance.index((min(distance)))] = 255
      test[test==-1] = 0

  test2 = label.copy()
  if index2 == -1:
      test2.fill(0)
  else:
      test2[test2!=index2] = -1
      test2[test2==index2] = 255
      test2[test2==-1] = 0
    

  y = color[label]
  y = np.uint8(y)

  res_from_clustering = np.bitwise_or(test2,test)

  # file_name_no_extension = os.path.splitext(file_name)[0]
      
  # cv2.imwrite("/content/drive/My Drive/BE Pro Materials/Diabetic Retinopath/Exudates/train_results/_candidate_exudates.bmp",edge_candidates)	
  # cv2.imwrite("/content/drive/My Drive/BE Pro Materials/Diabetic Retinopath/Exudates/train_results/_result_exudates_kmeans.bmp",y)	
  # cv2.imwrite("/content/drive/My Drive/BE Pro Materials/Diabetic Retinopath/Exudates/train_results/_test_result.bmp",test)
  # cv2.imwrite("/content/drive/My Drive/BE Pro Materials/Diabetic Retinopath/Exudates/train_results/_test2_result.bmp",test2)		
  final_candidates = np.bitwise_or(edge_candidates,res_from_clustering)	
  plt.imshow(final_candidates)
  OD_loc = gray_scale.copy()
  # cv2.circle(OD_loc,coordinates_OD, 70, (0,0,0), -10)
  # cv2.imwrite("/content/drive/My Drive/BE Pro Materials/Diabetic Retinopath/Exudates/train_results/_OD_.bmp",OD_loc)

  # cv2.imwrite("/content/drive/My Drive/BE Pro Materials/Diabetic Retinopath/Exudates/train_results/_final_candidate.bmp",final_candidates)
  # cl_res_dev = remove_bv_image("/content/drive/My Drive/BE Pro Materials/Diabetic Retinopath/Exudates/train_results/_final_candidate.bmp",bv_image_dash)
  # cv2.imwrite("/content/drive/My Drive/BE Pro Materials/Diabetic Retinopath/Exudates/train_results/removed_bv_from.bmp",cl_res_dev)
  # print("/content/drive/My Drive/BE Pro Materials/Diabetic Retinopath/Exudates/train_results/_final_candidates.bmp")		
  # final_candidates = np.bitwise_or(final_candidates,cl_res_dev)
  plt.imshow(final_candidates)
          
  # cv2.circle(final_candidates,coordinates_OD, 70, (0,0,0), -10)
#   maskk = cv2.imread("MASK.bmp")
  # final_candidates = np.bitwise_and(final_candidates,maskk[:,:,0])
      
  final_candidates = final_candidates.astype('uint8')
  final_candidates = cv2.dilate(final_candidates, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	
  plt.imshow(final_candidates,cmap='gray')	
  # cv2.imwrite("/content/drive/My Drive/BE Pro Materials/Diabetic Retinopath/Exudates/train_results/_final_candidates.bmp",final_candidates)
      
  candidates_vector = np.reshape(final_candidates,(final_candidates.size,1))/255
  print(final_candidates.shape,"SHAPE OF FINAL CANDIDATE")
  return edge_candidates
  
   
def preprocess_image(image,target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def index():
    return render_template("sample.html")

# @app.route("/prediction", methods=["POST"])
# def prediction():
#      global model
#      model = load_model('/home/ec2-user/environment/environment/wtproj/flaserve/model.hdf5')
#      print(".  Model Loaded  .")
#      message = request.get_json(force=True)
#      encoded = message['image']
#      decoded = base64.b64decode(encoded)
#      image = Image.open(io.BytesIO(decoded))
#      processed_image = preprocess_image(image, target_size=(299,299))
#      prediction = model.predict(processed_image).tolist()
#      print(prediction)
#      result = prediction[0].index(max(prediction[0]))
    
#      response={"prediction" : result}
#      return jsonify(response)

@app.route("/predict", methods=["POST"])
def predict():
    # global model
    # model = load_model('/home/ec2-user/environment/environment/wtproj/flaserve/model.hdf5')
    # print(".  Model Loaded  .")
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    # processed_image = preprocess_image(image, target_size=(299,299))
    
    # prediction = model.predict(processed_image, steps=6).tolist()
#     print(prediction)
    # result = str(prediction[0].index(max(prediction[0])))
    # image=cv2.resize(np.uint8(image),(3500,3500))
    arra=["hello"]
    # exu = str(disp(image))
    
    imagen=cv2.resize(np.uint8(image),(3500,3500))
    microan = str(dispma(imagen))
    imagenbv = str(dispbv(imagen))
    arra.append(exu)
    arra.append(imagenbv)
    arra.append(microan)
    # arra.append(result)
   
#     prediction = model.predict(processed_image).tolist()

    response = {
        'resu': {
            # '0': arra[0],
            'exudates': arra[1],
            'bloodves': arra[2],
            'microaneu': arra[3],
            # 'result':arra[4],
       
       
        }
    }
    return jsonify(response)



@app.route("/prediction", methods=["POST"])
def prediction():
    global model
    model = load_model('/home/ec2-user/environment/environment/wtproj/flaserve/model.hdf5')
    print(".  Model Loaded  .")
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(299,299))
    
    prediction = model.predict(processed_image).tolist()

    result = prediction[0].index(max(prediction[0]))
    

    response={"prediction" : result}
    

    return jsonify(response)

if __name__ == "__main__":
    # app.run(host=os.getenv('IP','0.0.0.0'),port=int(os.getenv('PORT','5000')))
    app.run(host="0.0.0.0", port="5000")
    app.debug=True
