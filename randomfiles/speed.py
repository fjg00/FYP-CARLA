
import cv2

def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor*2, fy=zoom_factor)
picture = cv2.imread("/home/justin/Downloads/FYP_icons/renders\/trial.jpg")

zoomm  =picture[30:170,70:130]
cv2.imwrite('/home/justin/Downloads/FYP_icons/renders\/trialf.jpg',zoomm)
cv2.imshow("qas",zoomm)
cv2.waitKey(0)
cv2.destroyAllWindows()