import cv2 as cv


# # Reading Images
my_image = cv.imread("faces.png")

gray = cv.cvtColor(my_image, cv.COLOR_BGR2GRAY)

def rescaleFrame(frame, scale=1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions)




resized_image = rescaleFrame(gray)
resized_color_image = rescaleFrame(my_image)

# cv.imshow('Gray Person Image', resized_image)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(resized_image, scaleFactor=1.1, minNeighbors=2)


print(f'{len(faces_rect)}')


for (x,y,w,h) in faces_rect:
    cv.rectangle(resized_color_image, (x,y), (x+w, y+h), (0,255,0), 2)


cv.imshow('Detected Faces', resized_color_image)

cv.waitKey(0)