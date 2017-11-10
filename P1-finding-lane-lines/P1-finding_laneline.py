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


def get_roi(img):
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


def draw_lines(img, lines, color, thickness):
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


def improved_lines(left_lines, right_lines, shape):
    ysize = shape[0]

    left_bottom = [left_lines[0][0], left_lines[0][1]]
    left_top = [left_lines[-1][2], left_lines[-1][3]]
    right_top = [right_lines[0][0], right_lines[0][1]]
    right_bottom = [right_lines[-1][2], right_lines[-1][3]]

    k_left = (left_top[1] - left_bottom[1]) * 1.0 / (left_top[0] - left_bottom[0])
    k_right = (right_top[1] - right_bottom[1]) * 1.0 / (right_top[0] - right_bottom[0])

    left_bottom2 = [int(left_bottom[0] - (left_bottom[1] - ysize) / k_left), ysize]
    left_top2 = [int(left_top[0] - (left_top[1] - (ysize / 2 + 50)) / k_left), ysize / 2 + 50]
    right_bottom2 = [int(right_bottom[0] - (right_bottom[1] - ysize) / k_right), ysize]
    right_top2 = [int(right_top[0] - (right_top[1] - (ysize / 2 + 50)) / k_right), ysize / 2 + 50]

    left_lines.append([left_bottom[0], left_bottom[1], left_bottom2[0], left_bottom2[1]])
    left_lines.append([left_top[0], left_top[1], left_top2[0], left_top2[1]])
    right_lines.append([right_bottom[0], right_bottom[1], right_bottom2[0], right_bottom2[1]])
    right_lines.append([right_top[0], right_top[1], right_top2[0], right_top2[1]])

    return left_lines, right_lines


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    img should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    left_lines, right_lines = divide_lines(img, lines)
    left_lines, right_lines = improved_lines(left_lines, right_lines, img.shape)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, left_lines, [255, 0, 0], 10)
    draw_lines(line_img, right_lines, [0, 255, 0], 10)

    return line_img


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)
    roi = get_roi(edges)
    lines = hough_lines(roi, 2, np.pi / 180, 15, 5, 20)
    result = cv2.addWeighted(image, 0.8, lines, 1., 0.)

    # plt.subplot(231), plt.imshow(image)
    # plt.title("origin"), plt.xticks([]), plt.yticks([])
    # plt.subplot(232), plt.imshow(gray, cmap='gray')
    # plt.title("gray"), plt.xticks([]), plt.yticks([])
    # plt.subplot(233), plt.imshow(edges, cmap='gray')
    # plt.title("edges"), plt.xticks([]), plt.yticks([])
    # plt.subplot(234), plt.imshow(roi, cmap='gray')
    # plt.title("roi"), plt.xticks([]), plt.yticks([])
    # plt.subplot(235), plt.imshow(lines, cmap='gray')
    # plt.title("lines"), plt.xticks([]), plt.yticks([])
    # plt.subplot(236), plt.imshow(result)
    # plt.title("result"), plt.xticks([]), plt.yticks([])
    # plt.show()
    return result

# image = mpimg.imread('test_images/solidWhiteCurve.jpg')
# process_image(image)

# subclip of the first 5 second from 0 to 5.
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

clip = VideoFileClip("test_videos/challenge.mp4")
output = "test_videos/challenge_result.mp4"
line_clip = clip.fl_image(process_image)
line_clip.write_videofile(output, audio=False)
