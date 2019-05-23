import numpy as np
import cv2 as cv
import speech_recognition as sr
from nltk.tokenize import word_tokenize
def vision():
	face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
	img = cv.imread('pic.jpg')
	img = cv.resize(img, (500, 500)) 
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if faces is not None:
		print("Face Detected")
	for (x,y,w,h) in faces:
	    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = img[y:y+h, x:x+w]
	    eyes = eye_cascade.detectMultiScale(roi_gray)
	    for (ex,ey,ew,eh) in eyes:
	        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	cv.imwrite('outputPic.jpg',img)
	cv.imshow('img',img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def audio():
	r = sr.Recognizer()
	audiosample = sr.AudioFile('Record-001.wav')
	with audiosample as source:
		audio = r.record(source)
	print("Speech Recorded")
	speechText = r.recognize_google(audio)
	print("Speech to Text: "+ speechText)
	f = open("speechToText.txt", "w")
	f.write(speechText)
	f.close()
def nlp():
	print("Natural Language Processing")
	print("Words in Speech to Text")
	f1 = open("speechToText.txt", "r")
	words = word_tokenize(f1.read())
	print(words)
	return(words) 
def input4():
	f1 = open("fragrance.txt", "r")
	f2 = open("output3.txt", "w")
	str = f1.read()
	f2.write(str)
	f1.close()
	f2.close()
	return
def input5(words):
	f1 = open("input4.txt", "r")
	f2 = open("output4.txt", "w")
	str = f1.read()
	inference = words[-1]+","+str
	print("Inference:" + inference)
	f2.write(inference)
	f1.close()
	f2.close()
	return
def main():
	vision()
	audio()
	words = nlp()
	input4()
	input5(words)
if __name__== "__main__":
  main()
