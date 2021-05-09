import cv2
import numpy as np
import face_recognition

#Converting a image from BRG to RGB because we get a image in BRG
ImgElon = face_recognition.load_image_file("Images/ElonMuskS.jpg")
ImgElon = cv2.cvtColor(ImgElon,cv2.COLOR_BGR2RGB)
ImgTest = face_recognition.load_image_file("Images/BillGates.jpg")
ImgTest = cv2.cvtColor(ImgTest,cv2.COLOR_BGR2RGB)

#Detecting the face
faceLoc = face_recognition.face_locations(ImgElon)[0]  #because we are giving Single image
#Encoding The face
encodeElon = face_recognition.face_encodings(ImgElon)[0]  #It will give us 4 coordinates(T,R,B,L)
#creating a rectangle using above coordinates
cv2.rectangle(ImgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)   #2 is thickness


faceLocTest = face_recognition.face_locations(ImgTest )[0]
encodeTest = face_recognition.face_encodings(ImgTest)[0]
cv2.rectangle(ImgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#Compare the images and it will take list
results = face_recognition.compare_faces([encodeElon],encodeTest)
#findind distance between faces
facedis = face_recognition.face_distance([encodeElon],encodeTest)
#Putting text on images
cv2.putText(ImgTest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow("Elon Musk",ImgElon)
cv2.imshow("Elon Test",ImgTest)
cv2.waitKey(0)