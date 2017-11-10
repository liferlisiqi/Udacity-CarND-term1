# **Finding Lane Lines on the Road** 

## Writeup Template


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. 

1. Converted the images to grayscale from RGB model 
    ![avatar][lanelines.jpg]
    [image1](/examples/grayscale.jpg) "Grayscale"
2. Use cv2.GaussianBlur() to blur the image

3. The first core operation: detect edges of a gray model image

4. After the edges have been got, my next step is to define a region of interest(i.e., ROI), this method is old but efficient. Cause the camere installed on the car is fixed, so the lane lines is in a specific region, usually a trapezoid.

5. Anothe core operation: hough transform edges to a set of lines represent by start point and end point. Hough transform get the image of lane lines we want.

6. Finally, we add the lane lines image and innitial image together. 




### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be my pipeline can't find lane lines exactly when there are too much sonshine or cross of light. Cause in grayscale model of the image, too many useless edges will be find as edges of lane lines by canny detection. Therefore, the real lane lines can't be divided from the noise edges by hough transform, which cause serious result.

Another shortcoming could be there are too many paraments have to be turn, maybe i can find a set of paraments fit one kind of environment, like straight lane lines under sunny day(the easiest case). However, no one can find all the paraments for all situations, or even all kinds of cases can't be enumerated.

And my pipeline cost too much time, which can not adapt to the real car running fast. 


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to find a more efficient and robust way to detect lane lines. Color space can be used to study how to find white and yellow more efficiently, and the way to fit lane lines can be update by polynomial.

Another potential improvement could be to use parallel algorithm to accelerate my pipeline, we all know Nvidia do good work at this space, thus use their technology is a better way.

