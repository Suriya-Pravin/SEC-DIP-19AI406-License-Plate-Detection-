# License plates Detection Project Exercise

## Name : Suriya Pravin M
## Reg. No: 212223230223



## Russian License Plate Blurring

Welcome to your object detection project! Your goal will be to use Haar Cascades to blur license plates detected in an image!

Russians are famous for having some of the most entertaining DashCam footage on the internet (I encourage you to Google Search "Russian DashCam"). Unfortunately a lot of the footage contains license plates, perhaps we could help out and create a license plat blurring tool?

OpenCV comes with a Russian license plate detector .xml file that we can use like we used the face detection files (unfortunately, it does not come with license detectors for other countries!)

----


#### 3 Ways to Approach this project:
* Just go for it! Use the image under the DATA folder called car_plate.jpg and create a function that will blur the image of its license plate. Check out the Haar Cascades folder for the correct pre-trained .xml file to use.
* Use this notebook! Here we offer a guide of what main steps you should take to complete the project.
* Jump to the solutions notebook and video to treat this entire project as code-along project where you can code along with us.

## Project Guide

Follow and complete the tasks below to finish the project!

**TASK: Import the usual libraries you think you'll need.**

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

**TASK: Read in the car_plate.jpg file from the DATA folder.**

```python
img = cv2.imread('car_plate.jpg')
```

**TASK: Create a function that displays the image in a larger scale and correct coloring for matplotlib.**

```python
def display(img):
    plt.figure(figsize=(10, 8))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
```

```python
display(img)
```

**TASK: Load the haarcascade_russian_plate_number.xml file.**

```python
plate_cascade = cv2.CascadeClassifier('haarcascade_licence_plate_rus_16stages.xml')

```

**TASK: Create a function that takes in an image and draws a rectangle around what it detects to be a license plate. Keep in mind we're just drawing a rectangle around it for now, later on we'll adjust this function to blur. You may want to play with the scaleFactor and minNeighbor numbers to get good results.**

```python
def detect_plate(img):
    plate_img = img.copy()
    plate_rects = plate_cascade.detectMultiScale(
        plate_img, 
        scaleFactor=1.1, 
        minNeighbors=3
    )

    print(f"Plates detected: {len(plate_rects)}")  # shows how many plates were found

    for (x, y, w, h) in plate_rects:
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (255, 0, 0), 5)

    return plate_img

```

```python
result = detect_plate(img)
```

```python
display(result)
```

**FINAL TASK: Edit the function so that is effectively blurs the detected plate, instead of just drawing a rectangle around it. Here are the steps you might want to take:**

1. The hardest part is converting the (x,y,w,h) information into the dimension values you need to grab an ROI (somethign we covered in the lecture 01-Blending-and-Pasting-Images. It's simply [Numpy Slicing](https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python), you just need to convert the information about the top left corner of the rectangle and width and height, into indexing position values.
2. Once you've grabbed the ROI using the (x,y,w,h) values returned, you'll want to blur that ROI. You can use cv2.medianBlur for this.
3. Now that you have a blurred version of the ROI (the license plate) you will want to paste this blurred image back on to the original image at the same original location. Simply using Numpy indexing and slicing to reassign that area of the original image to the blurred roi.

```python
def detect_and_blur_plate(img):
    plate_img = img.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in plate_rects:
        roi = plate_img[y:y+h, x:x+w]
        blurred_roi = cv2.medianBlur(roi, 25)
        plate_img[y:y+h, x:x+w] = blurred_roi

    return plate_img

```

```python
result = detect_and_blur_plate(img)
```

```python
display(result)
```

## Output
<Figure size 1000x800 with 1 Axes><img width="794" height="457" alt="image" src="https://github.com/user-attachments/assets/ef443de6-f350-4abf-be6d-d016b5dd53ae" />

<Figure size 1000x800 with 1 Axes><img width="794" height="457" alt="image" src="https://github.com/user-attachments/assets/d7ed9d63-2867-4448-9839-d6ea9dc19651" />

<Figure size 1000x800 with 1 Axes><img width="794" height="457" alt="image" src="https://github.com/user-attachments/assets/2385261a-c1ba-4aa0-bf83-88fcac0f5c5e" />

# Result:
The license plate in the input image was successfully detected using the Haar Cascade classifier. 
The detected plate region was localized by drawing a bounding box, and the system accurately 
identified the number plate area from the image.
