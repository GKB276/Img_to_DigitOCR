import cv2
import pytesseract

#72
#73

image = cv2.imread('100017873.jpg')
# image = cv2.imread("cropped_img\c9.jpg")
image = cv2.resize(image,(0,0),fx=2,fy=2)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (3, 3), 0)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


custom_config = r'--oem 3 --psm 6 outputbase digits'
# custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
text = pytesseract.image_to_string(thresh, config=custom_config)
cv2.imshow("Image",image)
print("Extracted Numbers:", text)
cv2.waitKey(0)
cv2.destroyAllWindows()
