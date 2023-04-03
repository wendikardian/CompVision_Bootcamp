import cv2

img = cv2.imread('photo.jpg')
print(img.shape)
img = cv2.GaussianBlur(img, (9, 9), 0)

green = (0, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)

cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), green, 3)
cv2.line(img, (0, 20), (img.shape[1], 20), red, 3)

cv2.rectangle(img, (300, 60), (450, 200), blue, 3)
cv2.putText(img, 'Person', (300, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 2)
cv2.circle(img, (200, 260), 100, (255, 255, 255), 3)
cv2.putText(img, 'Laptop', (200, 250), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)
