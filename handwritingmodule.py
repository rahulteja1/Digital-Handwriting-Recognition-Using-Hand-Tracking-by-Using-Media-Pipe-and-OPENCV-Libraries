import cv2
import time
import Handtrackingmodule as htm
import numpy as np
import os
import sys

overlayList = []  # list to store all the images

brushThickness = 5
#eraserThickness = 50
drawColor = (0, 0, 255)  # setting purple color
line = 0
xp, yp = 0, 0
imgCanvas = np.zeros((480, 640, 3), np.uint8)  # defining canvas

var_inits = False
mask = np.ones((480, 640))*255
mask = mask.astype('uint8')

# images in header folder
folderPath = "Header"
myList = os.listdir(folderPath)  # getting all the images used in code
# print(myList)
for imPath in myList:  # reading all the images from the folder    
	image = cv2.imread(f'{folderPath}/{imPath}')    
	if(image is None):
		continue
	overlayList.append(image)  # inserting images one by one in the overlayList
# print(overlayList)
header = overlayList[0]  # storing 1st image
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height
# sys.exit(1)
detector = htm.hand_detector()  # making object

while True:   
	# continue 
	# 1. Import image    
	success, img = cap.read()    
	# img = cv2.flip(img, 1)  # for neglecting mirror inversion    
	# 2. Find Hand Landmarks    
	img = detector.findhands(img)  # using functions fo connecting landmarks    
	lmList, bbox = detector.findposition(img,                                         draw=False)  # using function to find specific landmark position,draw false means no circles on landmarks    
	# sys.exit(1)
	if len(lmList) != 0:        
		print(lmList)        
		x1, y1 = lmList[8][1], lmList[8][2]  # tip of index finger        
		x2, y2 = lmList[12][1], lmList[12][2]  # tip of middle finger        
		
		# 3. Check which fingers are up        
		fingers = detector.fingersUp()        
		print(fingers)        
		# 
		# # 4. If Selection Mode - Two finger are up        
		if fingers[1] and fingers[2]:            
			xp, yp = 0, 0            
			# print("Selection Mode")            
			# checking for click            
			
			if y1 < 62:                
				if 0 < x1 < 50:                    
					drawColor = (0, 0, 255)                
				elif 50 < x1 < 100:                    
					drawColor = (255,0,200)                
				elif 100 < x1 < 150:                    
					drawColor = (0, 255, 0)                
				#elif 150 < x1 < 200:                    
					line = 4                
				elif 200 < x1 < 250:  # straight line                    
					# header = overlayList[0]                    
					line = 2                
				elif 250 < x1 < 300:  # rectangle                    
					# header = overlayList[1]                    
					line = 3                
				elif 300 < x1 < 350:  # circle                    
					# header = overlayList[2]                    
					line = 4                
				elif 350 < x1 < 400:  # line                    
					# header = overlayList[3]                    
					line = 1                
				elif 400 < x1 < 450:  # eraser                    
					# header = overlayList[3]                    
					drawColor = (0, 0, 0)            
			cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor,                          cv2.FILLED)  # selection mode is represented as rectangle        
		
		# 5. If Drawing Mode - Index finger is up        
		#line        
		if line == 1:            
			if fingers[1] and fingers[2] == False:                
				cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)                
				cv2.line(mask, (xp, yp), (x1, y1), drawColor, brushThickness)                
				cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)                
				xp, yp = x1, y1            
			else:                
				xp = x1                
				yp = y1        
		#straight line        
		elif line == 2:            
			if fingers[1] and fingers[2] == False:                
				cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)                
				if not(var_inits):                    
					xi, yi = x1, y1                    
					var_inits = True                
				cv2.line(img, (xi, yi), (x1, y1), drawColor, brushThickness)            
			else:                
				if var_inits:                    
					cv2.line(imgCanvas, (xi, yi), (x1, y1), drawColor, brushThickness)                    
					var_inits = False        
		#rectangle        
		elif line == 3:            
			if fingers[1] and fingers[2] == False:                
				cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)                
				if not(var_inits):                    
					xi, yi = x1, y1                    
					var_inits = True                
				cv2.rectangle(img, (xi, yi), (x1, y1), drawColor, brushThickness)            
			else:                
				if var_inits:                    
					cv2.rectangle(imgCanvas, (xi, yi), (x1, y1), drawColor, brushThickness)                    
					var_inits = False        
		#circle        
		elif line == 4:            
			if fingers[1] and fingers[2] == False:                
				cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)                
				if not(var_inits):                    
					xi, yi = x1, y1                    
					var_inits = True                
				cv2.circle(img, (xi, yi), int(((xi-x1)**2 + (yi-y1)**2)**0.5), drawColor, brushThickness)            
			else:                
				if var_inits:                    
					cv2.circle(imgCanvas, (xi, yi), int(((xi-x1)**2 + (yi-y1)**2)**0.5), drawColor, brushThickness)                    
					var_inits = False        
		#eraser        
		elif drawColor == (0, 0, 0):            
			eraserThickness = 50            
			if fingers[1] and fingers[2] == False:                
				#cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)                
				cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)                
				cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)               
				 #xp, yp = x1, y1            
			#else:               
				#xp = x1                
				#yp = y1        
		
		# merging two windows into one imgcanvas and img    
	
	# 1 converting img to gray    
	imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)    
	
	# 2 converting into binary image and thn inverting    
	_, imgInv = cv2.threshold(imgGray, 50, 255,                              cv2.THRESH_BINARY_INV)  # on canvas all the region in which we drew is black and where it is black it is cosidered as white,it will create a mask    
	
	imgInv = cv2.cvtColor(imgInv,                          cv2.COLOR_GRAY2BGR)  # converting again to gray bcoz we have to add in a RGB image i.e img    
	
	# add original img with imgInv ,by doing this we get our drawing only in black color    
	img = cv2.bitwise_and(img, imgInv)    
	
	# add img and imgcanvas,by doing this we get colors on img    
	img = cv2.bitwise_or(img, imgCanvas)    
	
	# setting the header image    
	img[0:62, 0:640] = header  # on our frame we are setting our JPG image acc to H,W of jpg images    
	
	cv2.imshow("Image", img)    
	cv2.imshow("Canvas", imgCanvas)    
	# cv2.imshow("Inv", imgInv)    
	cv2.waitKey()
