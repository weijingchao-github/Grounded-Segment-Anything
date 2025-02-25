import cv2

image_ori = cv2.imread("demo9.jpg")
image = cv2.resize(image_ori, (640, 480))
cv2.imwrite("demo9_resized.jpg", image)
