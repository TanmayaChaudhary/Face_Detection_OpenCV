import cv2 

myimage=cv2.imread(r"8.jpg")
myimage1=cv2.imread(r"8.jpg",cv2.IMREAD_GRAYSCALE)

#CascadeClassifier will tell if image is not
facedetector=cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
#it load haarcascade  for testing

face=facedetector.detectMultiScale(myimage1,1.2,5)   #1.3 is learning rate 30%
#print(face)
#for multiple faces in image
print("No. of face in my image")
print(len(face))
for x,y,w,h in face:
    cv2.rectangle(myimage,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.imshow("facedetected",myimage)
    
cv2.waitKey(0)    
cv2.destroyAllWindows()    