import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip


# cap = cv2.VideoCapture('/home/lsq/CarND-term1/CarND-LaneLines-P1/examples/'
#                        'P1_example.mp4')
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('lanelines', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


def region_of_interest(img):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    xsize = img.shape[1]
    ysize = img.shape[0]
    left_bottom = (0, ysize)
    left_top = (xsize / 2 - 50, ysize / 2 + 50)
    right_bottom = (xsize, ysize)
    right_top = (xsize / 2 + 50, ysize / 2 + 50)
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)


def line_vertexs(img, lines):
    xsize = img.shape[1]
    ysize = img.shape[0]
    x_middle = xsize / 2
    left_bottom = [0, 0]
    left_top = [0, ysize]
    right_bottom = [0, 0]
    right_top = [0, ysize]

    for line in lines:
        if abs(line[0][0] - line[0][2]) > 2:
            k = (line[0][3] - line[0][1]) * 1.0 / (line[0][2] - line[0][0])
            if line[0][0] < x_middle and k < 0:
                if line[0][1] > left_bottom[1]:
                    left_bottom = [line[0][0], line[0][1]]
                if line[0][3] < left_top[1]:
                    left_top = [line[0][2], line[0][3]]
            elif line[0][2] > x_middle and k > 0:
                if line[0][1] < right_top[1]:
                    right_top = [line[0][0], line[0][1]]
                if line[0][3] > right_bottom[1]:
                    right_bottom = [line[0][2], line[0][3]]

    if left_top[0] - left_bottom[0] < 0.00001:
        k_left = (left_top[1] - left_bottom[1]) * 1.0 / 0.00001
    else:
        k_left = (left_top[1] - left_bottom[1]) * 1.0 / (left_top[0] - left_bottom[0])

    if right_bottom[0] - right_top[0] < 0.00001:
        k_right = -(right_top[1] - right_bottom[1]) * 1.0 / 0.00001
    else:
        k_right = (right_top[1] - right_bottom[1]) * 1.0 / (right_top[0] - right_bottom[0])

    left_bottom = [int(left_top[0] - (left_top[1] - ysize) / k_left), ysize]
    right_bottom = [int(right_top[0] - (right_top[1] - ysize) / k_right), ysize]

    return left_top, left_bottom, right_top, right_bottom


def divide_lines(img, lines):
    x_middle = img.shape[1] / 2
    all_left_lines = []
    all_right_lines = []
    left_lines = []
    right_lines = []
    for line in lines:
        if abs(line[0][0] - line[0][2]) > 2:
            k = (line[0][3] - line[0][1]) * 1.0 / (line[0][2] - line[0][0])
            if line[0][0] < x_middle and k < -0.5:
                all_left_lines.append(line[0])
            elif line[0][2] > x_middle and k > 0.5:
                all_right_lines.append(line[0])
    all_left_lines.sort(key=lambda x: x[0])
    all_right_lines.sort(key=lambda x: x[0])
    for line in all_left_lines:
        if len(left_lines) != 0:
            if line[0] > left_lines[-1][2] and line[1] < left_lines[-1][3]:
                left_lines.append([left_lines[-1][2], left_lines[-1][3], line[0], line[1]])
                left_lines.append([line[0], line[1], line[2], line[3]])
        else:
            left_lines.append([line[0], line[1], line[2], line[3]])

    for line in all_right_lines:
        if len(right_lines) != 0:
            if line[0] > right_lines[-1][2] and line[1] > right_lines[-1][3]:
                right_lines.append([right_lines[-1][2], right_lines[-1][3], line[0], line[1]])
                right_lines.append([line[0], line[1], line[2], line[3]])
        else:
            right_lines.append([line[0], line[1], line[2], line[3]])

    return left_lines, right_lines


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    img should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # left_top, left_bottom, right_top, right_bottom = line_vertexs(img, lines)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # cv2.line(line_img, (left_bottom[0], left_bottom[1]), (left_top[0], left_top[1]), [255, 0, 0], 5)
    # cv2.line(line_img, (right_bottom[0], right_bottom[1]), (right_top[0], right_top[1]), [0, 255, 0], 5)

    left_lines, right_lines = divide_lines(img, lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, left_lines, [255, 0, 0], 10)
    draw_lines(line_img, right_lines, [0, 255, 0], 10)

    return line_img


def weighted_img(line_img, initial_img, alpha=0.8, beta=1., theta=0.):
    # result image = initial_img * alpha + img * beta + theta
    return cv2.addWeighted(initial_img, alpha, line_img, beta, theta)


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur_gray, 50, 200)
    interset_edges = region_of_interest(edges)
    hough_line = hough_lines(interset_edges, 2, np.pi / 180, 15, 40, 20)
    result = weighted_img(hough_line, image)
    #
    # plt.subplot(231), plt.imshow(image)
    # plt.title("origin"), plt.xticks([]), plt.yticks([])
    # plt.subplot(232), plt.imshow(gray, cmap='gray')
    # plt.title("gray"), plt.xticks([]), plt.yticks([])
    # plt.subplot(233), plt.imshow(edges, cmap='gray')
    # plt.title("edges"), plt.xticks([]), plt.yticks([])
    # plt.subplot(234), plt.imshow(interset_edges, cmap='gray')
    # plt.title("interset_edges"), plt.xticks([]), plt.yticks([])
    # plt.subplot(235), plt.imshow(hough_line, cmap='gray')
    # plt.title("hough_line"), plt.xticks([]), plt.yticks([])
    # plt.subplot(236), plt.imshow(result)
    # plt.title("result"), plt.xticks([]), plt.yticks([])
    # plt.show()
    return result


# image = mpimg.imread('/home/lsq/CarND-term1/CarND-LaneLines-P1/'
#                      'test_images/solidWhiteCurve.jpg')
# process_image(image)
clip = VideoFileClip("/home/lsq/CarND-term1/CarND-LaneLines-P1/"
                     "test_videos/challenge.mp4")
output = "/home/lsq/CarND-term1/CarND-LaneLines-P1/" \
         "test_videos/challenge_result.mp4"
line_clip = clip.fl_image(process_image)
line_clip.write_videofile(output, audio=False)
