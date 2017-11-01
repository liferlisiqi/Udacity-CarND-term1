import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip


def undistort():
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].transpose().reshape(-1, 2)

    # arrays to store object points and image points
    objpoints = []
    imgpoints = []

    images = glob.glob("camera_cal/cal*.jpg")
    for frame in images:
        img = cv2.imread(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    img = cv2.imread("camera_cal/calibration1.jpg")
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist


mtx, dist = undistort()


def region_of_interest(img):
    mask = np.zeros_like(img)

    xsize = img.shape[1]
    ysize = img.shape[0]
    left_bottom = (150, ysize)
    left_top = (xsize / 2 - 80, ysize / 2 + 100)
    right_bottom = (xsize - 50, ysize)
    right_top = (xsize / 2 + 80, ysize / 2 + 100)
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)

    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


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


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    left_lines, right_lines = divide_lines(img, lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, left_lines, [255, 0, 0], 10)
    draw_lines(line_img, right_lines, [0, 255, 0], 10)

    return line_img


def x_gradient(img):
    x_gra = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    edges = cv2.convertScaleAbs(x_gra)
    return edges


def brideye(img):
    xsize = img.shape[1]
    ysize = img.shape[0]
    left_bottom = (0, ysize)
    left_top = (xsize / 2 - 100, ysize / 2 + 100)
    right_bottom = (xsize, ysize)
    right_top = (xsize / 2 + 100, ysize / 2 + 100)
    pts1 = np.float32([left_top, right_top, left_bottom, right_bottom])
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    mtx = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, mtx, (500, 500))
    dst = x_gradient(dst)
    return dst


def process_image(image):
    udst = cv2.undistort(image, mtx, dist, None, mtx)
    hsv = cv2.cvtColor(udst, cv2.COLOR_RGB2HLS)
    sta = hsv[:, :, 2]
    blur = cv2.GaussianBlur(sta, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 200)
    edges = x_gradient(canny)
    roi = region_of_interest(edges)
    brid = brideye(roi)
    hough_line = hough_lines(edges, 2, np.pi / 180, 5, 40, 20)
    result = cv2.addWeighted(image, 0.8, hough_line, 1.0, 0.0)
    plt.subplot(231), plt.imshow(image)
    plt.title("origin"), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(sta, cmap='gray')
    plt.title("statu"), plt.xticks([]), plt.yticks([])
    plt.subplot(233), plt.imshow(edges, cmap='gray')
    plt.title("edges"), plt.xticks([]), plt.yticks([])
    plt.subplot(234), plt.imshow(roi, cmap='gray')
    plt.title("roi"), plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.imshow(brid, cmap='gray')
    plt.title("brid"), plt.xticks([]), plt.yticks([])
    plt.subplot(236), plt.imshow(result)
    plt.title("result"), plt.xticks([]), plt.yticks([])
    plt.show()
    return result


images = glob.glob("test_images/*.jpg")

for frame in images:
    image = cv2.imread(frame)
    process_image(image)

# clip = VideoFileClip("challenge_video.mp4")
# output = "challenge_result.mp4"
# line_clip = clip.fl_image(process_image)
# line_clip.write_videofile(output, audio=False)
