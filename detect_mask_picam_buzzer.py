#initialize buzzer, LEDs, Servo and Sendgrid

from gpiozero import Buzzer, LED
from time import sleep
import board
import adafruit_mlx90614
import base64
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (Mail, Attachment, FileContent, FileName, FileType, Disposition)
import RPi.GPIO as GPIO
from time import sleep
import RPi.GPIO as GPIO
from time import sleep



#Pin setup
GPIO.setmode(GPIO.BOARD)
GPIO.setup(3, GPIO.OUT)
pwm=GPIO.PWM(3, 50)
pwm.start(0)
GPIO.setwarnings(False)
buzzer = Buzzer(21)
red = LED(14)
green = LED(15)
i2c = board.I2C()
mlx = adafruit_mlx90614.MLX90614(i2c)


# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

def SetAngle(angle):
	duty = angle / 18 + 2
	GPIO.output(3, True)
	pwm.ChangeDutyCycle(duty)
	sleep(1)
	GPIO.output(3, False)
	pwm.ChangeDutyCycle(0)

def send_sms():
    message = Mail(
    from_email='songoku190189@gmail.com',
    to_emails='testhalkhabar@gmail.com',
    subject='Requirement Failure ',
    html_content='<strong>This person doesn"t meet the requirement.</strong>'
    )
    with open('image.jpg', 'rb') as f:
        data = f.read()
        f.close()
    encoded_file = base64.b64encode(data).decode()

    attachedFile = Attachment(
        FileContent(encoded_file),
        FileName('image.jpg'),
        FileType('image/jpg'),
        Disposition('attachment')
    )
    message.attachment = attachedFile
    

    sg = SendGridAPIClient('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')
    response = sg.send(message)

def SetAngle(angle):
	duty = angle / 18 + 2
	GPIO.output(3, True)
	pwm.ChangeDutyCycle(duty)
	sleep(1)
	GPIO.output(3, False)
	pwm.ChangeDutyCycle(0)

    
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector1K.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

def check():
    print('done')

c=0
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=500)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		temperature_thread = (mlx.object_temperature*9/5)+42
		if mask > withoutMask:
			color = (0, 255, 0)
			label = f'Mask On. Temp {(mlx.object_temperature*9/5)+42}'
			if (temperature_thread>98.5):
			    buzzer.on()
			    red.on()
			    c +=1
			    if c == 10:
			        cv2.imwrite('image.jpg',frame)
			        send_sms()
			        c=0
			else:
			    red.off()
			    green.on()
			    buzzer.off()
			    SetAngle(0)
                SetAngle(90)
                pvm.stop()
            GPIO.cleanup()
		else:
			label = f'Mask not Found. Temp {(mlx.object_temperature*9/5)+42}'
			c +=1
			if c == 10:
			    cv2.imwrite('image.jpg',frame)
			    send_sms()
			    c=0
			color = (0, 0, 255)
			buzzer.on()
			green.off()
			red.on()
		
		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX-50, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Face Mask Detector", frame)
	key = cv2.waitKey(1) & 0xFF
	
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
