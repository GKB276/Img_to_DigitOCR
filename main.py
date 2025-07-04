import cv2
import pytesseract

img = cv2.imread("pic.jpg")
img = cv2.resize(img,(0,0),fx=4,fy=4)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
blurred = cv2.GaussianBlur(otsu1, (5, 5), 0)
ret, otsu2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilated = cv2.dilate(otsu2, kernel, iterations=1)
custom_config = r'--oem 3 --psm 6 outputbase digits'
text = pytesseract.image_to_string(dilated,config=custom_config)

print("Extracted Text:\n", text)
cv2.imshow("Image", dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()
