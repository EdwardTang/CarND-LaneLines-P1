# **Finding Lane Lines on the Road** 


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

## Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.
Below are breakdown snapshots of my image pipline.
#### Phase 1: Image Preprocessing :
##### 1. Convert RGB image to HSV image
Reference: http://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html
[image1]: ./write_up_images/HSV.png "HSV"

![alt text][image1]
#### 2. Get binary mask for yellow and  white from the HSV image
[image2]: ./write_up_images/YWBinaryMask.png "Yellow White Binary Mask"

![alt text][image2]
 #### 3. Apply binary mask to HSV image
[image3]: ./write_up_images/YWBinaryMaskToBW.png "Yellow White Binary Mask in Black White"

![alt text][image3]
#### 4. Smooth the  binary masked image
[image4]: ./write_up_images/Blur.png "Smooth Binary Mask in Black White"

![alt text][image4]
#### 5. Apply canny detection on binary masked image
[image5]: ./write_up_images/canny.png "Canny Detection on binary masked image"

![alt text][image5]
#### 6. Apply Region of Interests to canny detection
[image6]: ./write_up_images/masked_canny.png "Apply Region of Interests to canny detection"

![alt text][image6]
#### 7.Draw masked canny detection in black and white.
[image7]: ./write_up_images/BWMaskedCanny.png "Apply Region of Interests to canny detection"

![alt text][image7]
#### 8.Apply Hough Tranformation and Line filtering to masked canny detection in black and white.
Then we got the first screen shot on the image tiles
[image8]: ./write_up_images/Screenshot1.png "Hough Tranformation and Line filtering"

![alt text][image8]
#### Phase 2: Lane Line Status Detection
As the status of lane line is dynamic,  we use ```Lane``` object's static method 
```python
@staticmethod    
def update_lane_with_filtered_segments(segments):
```
to update the Lane status. Noticed that we passed ```Line``` objects segments from Phase 1 to this funtion.

First, let's take a look at ```Line``` object. Each instance of ```Line``` represents a segment of a lane line (left or right). And it can also be any random line we got from the Hough Transformation. Thus, ```Line``` object in this project is defined to filter all the lines from Hough Transformation with the consideration of if the line is a candidate for being a segment of the lines of a lane. Based on the consensus, we use the following business logic:
``` python
@property
def is_candidate(self):
    """
    Business logic to check whether this hough line can be a candidate 
    for being a segment of a lane line.
    1. The line should not be horizontal and have a reasonable slope.
    2. The slop of line should be identical with the lane line's slope.
    3. The line should locate within the range of the lane line.
    4. The line should be below the vanishing point.
```

Given the candidates we got from the above logic, we can easily get rid of fixed Region of Interests.Though there is still noise mixed in the candidates, the candidates are close enough to the lane line we are looking for. However, the candidate logic relies heavily on detecting the differece between the slope of line and the slope of the lane. This can be improved in the future.

As for ```Lane``` object, it contains two lane lines, left and right. To update the status of Lane lines, I use the candidates from ```Line.is_candidate``` as points to fit 1st level polynomial with ```numpy.polyfit```

Another important issue we need to address is the stability of lane lines. To stable the lane lines, I used two methods listed as following:

1. **Memorizing N recent states of lane**: ```Lane``` object memorize N recent states of lane and update the memory with processing a new line state of current frame.
2. **Update the lane state with weighted memory**: As there is still noise mixed in the line candidates, we might easily get a wrong line extrapolation. To address this issue, first we need to differ unstable line states from stable ones. And then treat the unstable lines with different approach. Thus, if the estimated slope of the fitted line from the current frame is far different with the average of the memory, we flag this line is unstable. And then, we create a decision matrix to decide how to use current fitted line and memory's average to extrapolate the lane. The following shows how the decision matrix can be applied to line extrapolation:
```python 
    # A decision matrix for updating a lane line in order to keep it stable.
    # A weighted average of average lane position from memory and from the current frame.
    # 0.1 * frame position + 0.9 * avg from buffer: when the lane line is unstable.
    # 1 * frame position + 0 * memory: when the lane line is stable.
    DECISION_MATRIX = {False: [0.1,0.9], True: [1,0]}
```
```Lane``` object has a flag property to indicate the line's stability from the current frame: ```self.stable```. When the lane line is stable, the flag is ```True```, and region of interest is painted in milky white. While the line is lane line is unstable, the flag is ```False```, and the region of interest is painted in orange. In the video clip, you will see orange flashing in ROI when the lane line is unstable.

[image9]: ./write_up_images/unstableLine.png "When encounter unstable lane, ROI is orange. "

![alt text][image9]
[image10]: ./write_up_images/stableLine.png "When encounter stable lane, ROI is white. "

![alt text][image10]

##### Phase 3: Drawing 
For determining if the line is a candidate for being a segment of lane line, one of the conditions is that the position of the line should be below the vanishing point of the lane lines. To locate the vanishing point, the method ```update_vanishing_point```:
```python
@staticmethod
def update_vanishing_point(left, right):
    equation = left.coeffs - right.coeffs
    x = -equation[1] / equation[0]
    y = np.poly1d(left.coeffs)(x)
    x, y = map(int, [x, y])
    left.vanishing_point = [x, y]
    right.vanishing_point = [x, y]
```
which calculates the coordinates of the point where two lane lines intersect.  Moreover, a vanishing point also helps to draw the polygon that represents the region of interest.
### 2. Identify potential shortcomings with your current pipeline
 The first shortcoming is the ROI detection. When the slope of lane changes too fast, for instances, on moutain roads, the ROI detection will fail as it relies heavily on comparing the slope difference between current frame and the average of memory. The second shortcoming is the image binariztion. When the shadow on the lane is too heavy, the lane detection fails.
 
### 3. Suggest possible improvements to your pipeline
     1. I would say ROI detection. I can be better to identify a ROI automatically withou prededined a fixed Polygon.
     2. Better image binarization using deelp learning techniques which enable lane detection in various of conditions.e.g. Snow day.
