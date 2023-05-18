import cv2
path = '/home/justin/Desktop/parking_on.png'
image = cv2.imread(path)
print(image.shape)
image = cv2.resize(image,(140,120))
#cv2.imwrite("/home/justin/Downloads/FYP_icons/screen_radio.png",image)
cv2.imshow("",image)
cv2.waitKey(0)

