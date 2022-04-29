import cv2

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
face_trained_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# choose image for face-detector
img = cv2.imread('faces.png')


# converting img into grayscale
img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

# detect face
face_coordinates = face_trained_data.detectMultiScale(img_greyscale)
# print(face_coordinates)

# draw rectangle around face

# (x,y,w,h,)=face_coordinates detect one face only
# (x,y,w,h,)=face_coordinates[1] by increasing number of array equal to detection of faces
# using for loop is a dynamic approach to detect faces,for example: if in image are 2 two face the AI detect only faces and viceversa

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y),  (x+w, y+h), (0, 255, 0), 2)
    # cv2.circle(img, (x, y,), (44), (0, 255, 0), 2) to be improved


cv2.imshow('qll in one faces', img)

cv2.waitKey()
