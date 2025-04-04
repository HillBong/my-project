import io
import os
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

PURE_BAND_TYPE = ['mp', 'imt', 'graystrip']
GRAY_STRIP_JUDGE_DIRECTION_METHOD = ['grayblock', 'number']
CUT_TYPE = ['cut1', 'cut2', 'cut3', 'cut4']
ALL_PATHS = {}
CONFIG = {}


def base_path(path):
    if getattr(sys, 'frozen', None):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)


# 灰度图片转二值化
# 局部阈值法,大津法
def binary_local_picture(image, threshold_value):
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, threshold_value, 0.1)
    return binary


# 全局阈值
def binary_global_picture(image, threshold_value):
    if isinstance(threshold_value, int):
        _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


# 写图片
def write_picture(img, path_to_write):
    cv2.imencode('.png', img)[1].tofile(path_to_write)


def count_x_black(image):
    count_x_black = []
    for x in image.T:
        count_x_black.append(np.count_nonzero(x == 0) / image.shape[0])
    return count_x_black


def show_picture(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# show多个图片不定个数
def show_multi_picture(images):
    i = 0
    for img in images:
        if img is not None:
            cv2.namedWindow('image_{}'.format(i), cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow('image_{}'.format(i), img)
            i += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_imt_baseline_begin_position(digital_position_result, black_position_result):
    """
    思路：设定一个初始值230
    最右端 350
    最左端 200
    首先向右找，如果找到>0.8的且超过4个的就是
    找不到向左边找
    :param digital_position_result
    :param black_position_result
    :return:
    """
    if (not isinstance(digital_position_result, tuple)) or len(digital_position_result) != 3 or \
            (not isinstance(black_position_result, list)) or len(black_position_result) == 0:
        return 0
    right_position = 500
    i = int(np.around(digital_position_result[1]))
    for black in black_position_result:
        if black[0] > i and black[2] > 0.3:
            return int(np.around(black[0]))
        if black[0] > right_position:
            break
    return 0


def get_overlapping_area_coverage_rate(x1, x2, x3, x4):
    """
    x1、x2真实坐标位置
    x3、x4大概坐标位置
    :param x1:
    :param x2:
    :param x3:
    :param x4:
    :return:
    """
    max_left = max(x1, x3)
    min_right = min(x2, x4)
    min_width = x2 - x1
    return (min_right - max_left) / min_width


def set_imt_tip(tip, j, left, right, image_rotate_average_gray):
    """
    给mp条带设值是否有黑块
    :param tip:
    :param j:
    :param left:
    :param right:
    :param image_rotate_average_gray:
    :return:
    """
    width = right - left
    min_average_gray = np.min(image_rotate_average_gray[left: right])
    # 取周围+-50以内的最大值避免干扰
    left = 0 if left - 50 else left - 50
    right = len(image_rotate_average_gray) if right + 50 > len(image_rotate_average_gray) else \
        right + 50
    max_average_gray = np.max(image_rotate_average_gray[left: right])
    # hiv-2特殊处理
    if j == 10:
        if width < 17:
            return
    if max_average_gray - min_average_gray < 15:
        tip[j] = '+'
    else:
        tip[j] = '+'


# 膨胀操作
def dilate_picture(image, kernel_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(image, kernel, iterations)


# 开操作
def open_picture(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


# 关操作
def close_picture(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def crop_effective_part_picture(img_binary, img):
    """
    第二次裁剪将无效部分去除掉
    第一个参数是二值化图
    第二个参数是需要裁剪的图片
    :return:
    """
    if img is None or (not isinstance(img, np.ndarray)) or len(img.shape) < 2 or img.shape[0] == 0 or img.shape[1] == 0:
        return img_binary, img
    loc = np.where(img_binary > 0)
    lx = []
    ly = []
    bx = sx = sy = by = 0
    for pt in zip(*loc[::-1]):
        lx.append(pt[1])
        ly.append(pt[2])
    if len(lx) != 0:
        sx = min(lx)
        bx = max(lx)
    if len(ly) != 0:
        sy = min(ly)
        by = max(ly)
    return img_binary[sy:by, sx:bx], img[sy:by, sx:bx]


def crop_effective_part_picture_gray_y(img_binary, img, image_gray_rotate):
    """
    将y轴上多余的裁掉
    第一个参数是二值化图
    第二个参数是需要裁剪的彩色图片
    第三个参数是需要裁剪的灰度图片
    :return:
    """
    if img is None or (not isinstance(img, np.ndarray)) or len(img.shape) < 2 or img.shape[0] == 0 or img.shape[1] == 0 \
            or image_gray_rotate is None:
        return img_binary, img, image_gray_rotate
    loc = np.where(img_binary > 0)
    ly = []
    sy = by = 0
    for pt in zip(*loc[::-1]):
        ly.append(pt[2])
    if len(ly) != 0:
        sy = min(ly)
        by = max(ly)
    return img_binary[sy:by, :], img[sy:by, :], image_gray_rotate[sy:by, :]


def crop_effective_part_picture_y(image_global, image_global_crop):
    """
    第四次裁剪将多余的y轴上的无效部分去除掉,
    裁剪的原则是黑色区域大于x轴上黑色像素总和最小值的1.3倍就丢弃
    从左下角到右上角或者从右上角到左下角
    :param image_global:
    :param image_global_crop:
    :return:
    """
    y = count_y_black_from_corn(image_global)
    # y = count_y_black(image_global)
    y_start = 0
    y_end = len(y) - 1
    # y_threshold = np.min(y) * 3
    while y_start < image_global.shape[0] // 4 and y[y_start] > image_global.shape[1] // 4:
        y_start += 1
    while y_end > image_global.shape[0] * 3 // 4 and y[y_end] > image_global.shape[1] // 4:
        y_end -= 1

    return image_global[y_start: y_end, :], image_global_crop[y_start: y_end, :]


def get_sv_image(image, crop_original_gray_binary_image):
    """
    取得彩色图片SV通道相加的图片
    :param crop_original_gray_binary_image:
    :param image:
    :return:
    """
    if len(image.shape) != 3:
        return image
    crop_hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # show_multi_picture(
    #     (image, image[:, :, 0], image[:, :, 1], image[:, :, 2]))
    crop_yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    show_multi_picture(
        (image, crop_yuv_image, crop_yuv_image[:, :, 0], crop_yuv_image[:, :, 1], crop_yuv_image[:, :, 2]))
    sv_image = crop_hsv_image[:, :, 1].astype(np.int64) + crop_hsv_image[:, :, 2].astype(np.int64)
    sv_image = np.where(sv_image > 255, 255, sv_image).astype(np.uint8)
    sv_binary_image = binary_global_picture(sv_image, None)
    sv_binary_image_close_1 = close_picture(sv_binary_image, 1)
    sv_binary_open_25_image = open_picture(sv_binary_image_close_1, 35)
    sv_binary_open_25_close_25_image = close_picture(sv_binary_open_25_image, 25)
    show_multi_picture(
        (image, crop_hsv_image, crop_hsv_image[:, :, 0], crop_hsv_image[:, :, 1], crop_hsv_image[:, :, 2], sv_image,
         sv_binary_image, sv_binary_open_25_image, sv_binary_open_25_close_25_image, crop_original_gray_binary_image))
    return sv_binary_open_25_image.reshape((*sv_binary_image.shape, 1))


def count_y_black_from_corn(image):
    count_y_black = []
    for x in image:
        i = 0
        length = x.shape[0]
        left = 0
        right = 0
        # 从左边到右边一次
        while i < length:
            if x[i] == 255:
                break
            left += 1
            i += 1
        i = length - 1
        while i >= 0:
            if x[i] == 255:
                break
            right += 1
            i -= 1
        count_y_black.append(max(left, right))
    return count_y_black


def flip_180(image):
    image = np.flipud(image)
    image = np.fliplr(image)
    return image


def get_y_average_gray(img):
    if img is None or len(img) == 0:
        return []
    y_average_gray = []
    width = img.shape[1]
    for i in range(width):
        y_average_gray.append(np.mean(img[:, i]))
    return y_average_gray


blot_check_x_y = [450, 1300]


def get_blot_limit(blot):
    x = blot[blot_check_x_y[0], :]
    y = blot[:, blot_check_x_y[1]]
    x_start = 0
    x_end = x.shape[0] - 1
    y_end = 0
    while x_start < x.shape[0] and x[x_start] > 120:
        x_start += 1
    while x_end > 0 and x[x_end] > 120:
        x_end -= 1
    while y_end < y.shape[0] and y[y_end] > 120:
        y_end += 1
    return [x_start, x_end, 0, y_end]


def save_average_gray_image(y_average_gray, image, path_to_average_gray):
    if y_average_gray is None or len(y_average_gray) < 1:
        return
    fig = plt.figure(figsize=(25, 5))
    plt.ylim(0, np.uint8(np.max(y_average_gray)) + 5)
    plt.xlim(0, len(y_average_gray))
    plt.minorticks_on()
    plt.tick_params(which='both', direction='in', width=2.2, bottom=True, top=True, left=True,
                    right=True, length=10, labelsize=25)
    plt.tick_params(which='major', length=20)
    plt.yticks(range(0, 255, 50))
    plt.xticks(range(0, len(y_average_gray), 500), color='w')
    plt.grid(True)
    plt.plot(range(len(y_average_gray)), y_average_gray, 'k', linewidth=2.2)
    plt.xlabel("Distance (pixels)", fontsize=25, color='w')  # 横坐标名字
    plt.ylabel("GrayValue", fontsize=25)  # 纵坐标名字
    plt.tight_layout()
    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_jpg(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    blot = Image.open(buffer)
    blot_temp = np.asarray(blot)[:, :, 0]
    blot = np.ones((blot_temp.shape[0] + 80, blot_temp.shape[1])) * 255
    blot[100:, :] = blot_temp[:-20, :]
    blot_limit = get_blot_limit(blot)
    image_resize = cv2.resize(image, (blot_limit[1] - blot_limit[0], blot_limit[3] - blot_limit[2]),
                              interpolation=cv2.INTER_NEAREST)
    blot[blot_limit[2]: blot_limit[3], blot_limit[0]: blot_limit[1]] = image_resize
    write_picture(blot, path_to_average_gray)
    plt.close('all')
    buffer.close()


def rotate_picture(img, img_binary):
    """
    将图片旋转到水平位置
    :param img_binary:
    :param img: 要旋转的图片
    :param hive_type: 条带类型
    :return: 旋转后端的图片
    """
    image_result = img.copy()
    # 边缘检测
    # cv2.imshow("0", cv2.resize(img_binary, (1920, 240)))
    edges = cv2.Canny(img_binary, 30, 60, apertureSize=3)
    # cv2.imshow("1", cv2.resize(edges, (1920, 240)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 膨胀操作,强化边缘检测的效果，同时将断掉的线连接起来
    edges = dilate_picture(edges, 3, 2)
    # cv2.imshow("2", cv2.resize(edges, (1920, 240)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, 200, 500, 100, 200)
    edges_vis = np.zeros_like(img)
    # print(lines)
    # print(len(lines))
    # exit()
    for x in lines:
        x1, y1, x2, y2 = x[0]
        edges_vis = cv2.line(edges_vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # print(lines)
    # print(len(lines))
    # cv2.imshow("3", cv2.resize(edges_vis, (1920, 240)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    i = 1
    # 对通过霍夫变换得到的数据进行遍历
    results = []
    # a = np.zeros_like(img_binary)
    horizontal_line_point = []
    if lines is None:
        return binary_global_picture(image_result, None), image_result
    for line in lines:
        x1, y1, x2, y2 = line[0]  # 两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
        if pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5) < 350:
            continue
        # cv2.line(a, (x1, y1), (x2, y2), (255, 255, 255), 2)  # 在原图上画线
        # 转换为浮点数，计算斜率
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)
        # print("x1=%s,x2=%s,y1=%s,y2=%s" % (x1, x2, y1, y2))
        if x2 - x1 == 0:
            # print("直线是竖直的")
            result = 90
        elif y2 - y1 == 0:
            # print("直线是水平的")
            result = 0
            horizontal_line_point.append((x1, x2, y1, y2))
        else:
            # 计算斜率
            k = -(y2 - y1) / (x2 - x1)
            # 求反正切，再将得到的弧度转换为度
            result = np.arctan(k) * 57.29577
            # print("直线倾斜角度为：" + str(result) + "度")
        results.append(result)
        i = i + 1
    # 因为各种条带的图像质量不一样，有的条带质量很高，容易矫正，有的条带质量很差，经常出现直线检测的时候，直线断联、直线很短、检测到的上下直线斜率不一的情况
    # 以下是经验代码，可以参考学习，但不作为重点要求

    # 只有水平线超过3跟的才可以使用这种方式求斜率
    # 对水平线进行分组，所有y轴相差不到3的都分为一组
    result1 = 0
    result2 = 0
    if len(horizontal_line_point) > 3:
        horizontal_line_point.sort(key=lambda x: x[2])
        horizontal_line_point_group = []
        temp = [horizontal_line_point[0]]
        for index in range(1, len(horizontal_line_point)):
            if abs(horizontal_line_point[index][2] - horizontal_line_point[index - 1][2]) > 3:
                horizontal_line_point_group.append(temp)
                temp = []
            temp.append(horizontal_line_point[index])
        if len(temp) > 0:
            horizontal_line_point_group.append(temp)
        # 将线的条数最多的两组取出来做为上下线
        if len(horizontal_line_point_group) > 1:
            horizontal_line_point_group.sort(key=lambda x: len(x), reverse=True)
            # 求上下两根线的斜率
            # 最左端
            left_point_1 = min(horizontal_line_point_group[0], key=lambda x: min(x[0], x[1]))
            left_point_2 = min(horizontal_line_point_group[1], key=lambda x: min(x[0], x[1]))
            # 最右端
            right_point_1 = max(horizontal_line_point_group[0], key=lambda x: max(x[0], x[1]))
            right_point_2 = max(horizontal_line_point_group[1], key=lambda x: max(x[0], x[1]))
            k1 = -(right_point_1[2] - left_point_1[2]) / (
                    max(right_point_1[0], right_point_1[1]) - min(left_point_1[0], left_point_1[1]))
            k2 = -(right_point_2[2] - left_point_2[2]) / (
                    max(right_point_2[0], right_point_2[1]) - min(left_point_2[0], left_point_2[1]))
            result1 = np.arctan(k1) * 57.29577
            result2 = np.arctan(k2) * 57.29577

    results = sorted(results)
    angle = 0
    if len(results) > 2:
        # 中间取一个，两头分别取两个
        angle += results[0] + results[1] + results[len(results) // 2] + results[-1] + results[-2]
        angle /= 5
    elif len(results) == 2:
        angle = results[0] + results[1]
        angle / 2
    else:
        angle = 0 if len(results) == 0 else results[0]

    if abs(angle) < abs(result1 + result2) / 2:
        coefficient = 1.4
    else:
        coefficient = 1.4
    if len(horizontal_line_point) <= 3:
        coefficient = 1.0
    if not (angle < 0 and result1 == 0 and result2 == 0):
        temp_angle = (0 if angle == 0 else 1) + (0 if result1 == 0 else 1) + (0 if result2 == 0 else 1)
        angle = (angle * coefficient + result1 + result2) / (1 if temp_angle == 0 else temp_angle)

    # 通过斜率对图片进行矫正
    M = cv2.getRotationMatrix2D((image_result.shape[1] // 2, image_result.shape[0] // 2), -angle * 1.1, 1.0)
    rotated = cv2.warpAffine(image_result, M, (image_result.shape[1], image_result.shape[0]))
    # cv2.imshow("7", cv2.resize(image_result, (1920, 240)))
    # cv2.imshow("8", cv2.resize(rotated, (1920, 240)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return binary_global_picture(rotated, None), rotated


def handle_rotate_picture1(image_rotate):
    # cv2.imshow("0", cv2.resize(image_rotate, (1920, 240)))
    image_global_binary = binary_global_picture(image_rotate, None).reshape(*image_rotate.shape, 1)
    # cv2.imshow("1", cv2.resize(image_global_binary, (1920, 240)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 多次矫正
    image_rotate_global_binary, image_rotate = rotate_picture(image_rotate, image_global_binary)
    # cv2.imshow("1", cv2.resize(image_rotate_global_binary, (1920, 240)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image_rotate_global_binary_open = open_picture(image_rotate_global_binary, 15).reshape(
        *image_rotate.shape, 1)
    # cv2.imshow("2", cv2.resize(image_rotate_global_binary_open, (1920, 240)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image_rotate_global_binary_crop, image_rotate_crop = crop_effective_part_picture(
        image_rotate_global_binary_open, image_rotate)
    # cv2.imshow("3", cv2.resize(image_rotate_crop, (1920, 240)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image_rotate_crop, image_rotate_global_binary_crop


def handle_rotate_picture(image_rotate):
    image_global_binary = binary_global_picture(image_rotate, None).reshape(*image_rotate.shape, 1)
    # 多次矫正
    # show_multi_picture((image_rotate, image_global_binary))

    image_rotate_global_binary, image_rotate = rotate_picture(image_rotate, image_global_binary)
    image_rotate_global_binary = image_rotate_global_binary.reshape(*image_rotate.shape, 1)
    average_gray_begin = np.mean(image_rotate[:, :50])
    average_gray_after = np.mean(image_rotate[:, -50:])
    if average_gray_begin > 50 and average_gray_after > 50:
        return image_rotate, image_rotate_global_binary[:, :]
    image_rotate_global_binary_open = open_picture(image_rotate_global_binary, 15).reshape(
        *image_rotate.shape, 1)
    # show_multi_picture((image_rotate, image_rotate_global_binary_open))
    image_rotate_global_binary_crop, image_rotate_crop = crop_effective_part_picture(
        image_rotate_global_binary_open, image_rotate)
    return image_rotate_crop, image_rotate_global_binary_crop


def binary_global_picture_picture_1(image, rate, threshold):
    """
    将条带分开二值化，按照从前往后的比例
    :param image:
    :param threshold:
    :param rate
    :return:
    """
    if image is None or (not isinstance(image, np.ndarray)) or len(image.shape) < 2 or image.shape[0] == 0 or \
            image.shape[1] == 0:
        return None
    if rate > 1.0:
        rate = 1.0
    if rate < 0:
        rate = 0
    length = int(image.shape[1] * rate)
    image_left = image[:, : length]
    image_right = image[:, length:]
    image_left_binary = binary_global_picture(image_left, threshold)
    image_right_binary = binary_global_picture(image_right, threshold)
    return np.concatenate((image_left_binary, image_right_binary), axis=1)


def crop_rotate_image(crop_original_image):
    image_rotate_crop = crop_original_image.copy()
    # cv2.imshow("0", cv2.resize(image_rotate_crop, (1920, 240)))

    # 1. 第一次裁剪矫正、横向矫正
    image_rotate_crop, image_rotate_global_binary_crop = handle_rotate_picture1(image_rotate_crop)
    # cv2.imshow("1", cv2.resize(image_rotate_crop, (1320, 140)))

    # cv2.imshow("1", cv2.resize(image_rotate_crop, (1320, 140)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 条带高度阈值
    height_threshold = 90
    # 2. 第二次裁剪矫正、纵向矫正调整
    while image_rotate_crop.shape[0] > height_threshold:
        # 高度
        width = image_rotate_global_binary_crop.shape[0] // 2
        if np.mean(image_rotate_global_binary_crop[:width, :]) > np.mean(
                image_rotate_global_binary_crop[-width:, :]):
            image_rotate_crop = image_rotate_crop[:-10, :]
        else:
            image_rotate_crop = image_rotate_crop[10:, :]
        image_rotate_global_binary_crop = binary_global_picture(image_rotate_crop, None). \
            reshape(*image_rotate_crop.shape, 1)

        image_rotate_global_binary_crop = open_picture(image_rotate_global_binary_crop, 5).reshape(
            *image_rotate_crop.shape, 1)

        image_rotate_global_binary_crop, image_rotate_crop = crop_effective_part_picture(
            image_rotate_global_binary_crop, image_rotate_crop)
    # cv2.imshow("2", cv2.resize(image_rotate_crop, (1320, 140)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 3. 第三次裁剪矫正、横向矫正、微微调整
    for i in range(2):
        image_rotate_crop, image_rotate_global_binary_crop = handle_rotate_picture(image_rotate_crop)
    # cv2.imshow("3", cv2.resize(image_rotate_crop, (1320, 140)))
    # cv2.imshow("3.1", cv2.resize(image_rotate_global_binary_crop.reshape(*image_rotate_crop.shape), (1320, 140)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 4. 第四次裁剪矫正、纵向矫正、微微调整
    image_rotate_global_binary_crop_xy, image_rotate_crop_xy = crop_effective_part_picture_y(
        image_rotate_global_binary_crop.reshape(*image_rotate_crop.shape), image_rotate_crop)
    if image_rotate_crop_xy is None or len(image_rotate_crop_xy) == 0:
        return None
    # cv2.imshow("4", cv2.resize(image_rotate_crop_xy, (1320, 140)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image_rotate_crop_xy


def object_detection_predict(image, model):
    """
    使用深度学习目标检测图片
    """
    if image is None or not isinstance(image, np.ndarray) or len(image) == 0:
        return None
    image = image.astype('float32')
    return model.predict(image)


def get_object_detection_result(result):
    """
    获取目标检测算法的结果包括识别数字和识别黑块
    :param result:
    :return:
    """
    if result is None or not isinstance(result, np.ndarray) or len(result) == 0:
        return [], []
    digital_result, black_result = [], []
    for i in result:
        if int(i[5]) == 0:
            digital_result.append(i)
        else:
            black_result.append(i)
    return digital_result, black_result


def get_object_seg_result(result):
    """

    :param result:
    :return:
    """
    if result is None or not isinstance(result, dict) or len(result) == 0:
        return np.array([])
    return np.where(result.get('label_map') == 0, 0, 255).astype(np.uint8)


def order_points(pts):
    # pts为轮廓坐标
    # 列表中存储元素分别为左上角，右上角，右下角和左下角
    rect = np.zeros((4, 2), dtype="float32")
    # 左上角的点具有最小的和，而右下角的点具有最大的和
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算点之间的差值
    # 右上角的点具有最小的差值,
    # 左下角的点具有最大的差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 返回排序坐标(依次为左上右上右下左下)
    return rect.astype(np.int32)


def get_slope_frm_min_rect(pts):
    """
    从最小外接矩形中获取斜率
    :param pts:
    :return:
    """
    rect = order_points(pts)
    k1 = -(rect[0][1] - rect[1][1]) / (rect[0][0] - rect[1][0])
    k2 = -(rect[2][1] - rect[3][1]) / (rect[2][0] - rect[3][0])
    result1 = np.arctan(k1) * 57.29577
    result2 = np.arctan(k2) * 57.29577
    return (result1 + result2) / 2


def recognize_digital(digital_result, black_result, length, score=0.19):
    """
    识别数字的结果，
        1. 如果数字的最大得分小于0.6就判定为太黑，返回0(备选方案)
            1. 如果数字在左端返回1，判断条件位于数字所在位置左边的黑块小于位于右边的黑块数，left_count，不需要翻转
            2. 如果数字在右端返回2，判断条件位于数字所在位置左边的黑块大于位于右边的黑块数，right_count，需要翻转
        2. 数字所在位置位于前一半就说明不需要翻转返回1
    :param digital_result:
    :param black_result:
    :param length:
    :param score
    :return:
    """
    if len(digital_result) == 0 or length == 0:
        return 0, None
    max_score_digital = max(digital_result, key=lambda x: x[4])
    if max_score_digital[4] < score:
        black_result = sorted(black_result, key=lambda x: x[0])
        # 如果算法没识别到数字，可能是当作黑块处理了，在这里打个补丁，用来判断，如果前后端少于180个像素点位置找到分数比较高的黑块很有可能是数字
        left_black_result = list(filter(lambda x: x[0] < 180 and x[4] > 0.3, black_result))
        right_black_result = list(filter(lambda x: x[2] > length - 180 and x[4] > 0.3, black_result))
        if len(left_black_result) == len(right_black_result) == 0:
            return 0, None
        if len(left_black_result) >= len(right_black_result):
            max_score_digital = max(left_black_result, key=lambda x: x[4])
            return 1, (max_score_digital[0], max_score_digital[2], max_score_digital[4])
        else:
            max_score_digital = max(right_black_result, key=lambda x: x[4])
            return 2, (length - max_score_digital[2], length - max_score_digital[0], max_score_digital[4])
    digital_position = (max_score_digital[0] + max_score_digital[2]) / 2
    # 位于左边
    if digital_position < length / 4:
        return 1, (max_score_digital[0], max_score_digital[2], max_score_digital[4])
    # 位于右边
    elif digital_position > length / 4 * 3:
        return 2, (length - max_score_digital[2], length - max_score_digital[0], max_score_digital[4])
    # 检测有误
    else:
        return 0, None


def get_position_result(digital_flag, black_result, image_rotate):
    """
    对黑块位置进行处理，并且将需要翻转的条带翻转过来
    :param digital_flag:
    :param black_result:
    :param image_rotate:
    :return:
    """
    image = image_rotate.copy()
    black_position_result = []
    length = image_rotate.shape[1]
    if digital_flag == 1:
        for i in black_result:
            black_position_result.append((i[0], i[2], i[4]))
    else:
        for i in black_result:
            black_position_result.append((length - i[2], length - i[0], i[4]))
        image = flip_180(image)
    black_position_result.sort(key=lambda x: x[0])
    return black_position_result, image


# 设置黑块过于黑但是检测算法没有找到的


def set_mp_tip(tip, j, left, right, image_rotate_average_gray, image_average_gray, black_name):
    """
    给mp条带设值是否有黑块
    :param tip:
    :param j:
    :param left:
    :param right:
    :param image_rotate_average_gray:
    :param black_name:
    :param image_average_gray
    :return:
    """
    min_average_gray = np.min(image_rotate_average_gray[left: right])
    width = right - left
    left_ = left
    right_ = right
    max_average_gray_ = np.max(image_rotate_average_gray[left_: right_])
    # 取周围+-50以内的最大值避免干扰
    left = 0 if left - 50 else left - 50
    right = len(image_rotate_average_gray) if right + 50 > len(image_rotate_average_gray) else \
        right + 50
    max_average_gray = np.max(image_rotate_average_gray[left: right])
    # 给HIV-2打给补丁
    if 'HIV-2' == black_name and (max_average_gray - min_average_gray) / width < 0.5:
        return
    if max_average_gray - min_average_gray < 15:
        tip[j] = '+'
    else:
        tip[j] = '+'
    if tip[j] == '+':
        return

    # 使用原始值再次检验下
    min_average_gray_ = np.min(image_average_gray[left_: right_])
    max_average_gray_ = np.max(image_average_gray[left_: right_])
    if min_average_gray_ < 7:
        tip[j] = '+'
        return
    if max_average_gray_ - min_average_gray_ < 15:
        tip[j] = '+'
    else:
        tip[j] = '+'


def set_mp_p17_tip_and_distance(MpBand, black_position_result, tip, judge_point_distance, image_rotate_average_gray,
                                image_average_gray):
    filter_by_area_cover = list(filter(lambda x: get_overlapping_area_coverage_rate(
        x[0], x[1], MpBand.SPEC_POINT_RANGE['p17'][0], MpBand.SPEC_POINT_RANGE['p17'][1]) > 0.6, black_position_result))
    # 再找分数最大的
    if len(filter_by_area_cover) == 0:
        return
    # 取分数最高的三个
    max_score_position = sorted(filter_by_area_cover, key=lambda x: x[2], reverse=True)[:3]
    if max_score_position[0][2] < 0.4:
        return
    # 对黑块按照x轴排序
    max_score_position = sorted(max_score_position, key=lambda x: x[0])
    if len(max_score_position) == 1:
        max_score_position = max_score_position[0]
    elif len(max_score_position) == 2:
        # 如果有两个就取像素范围内偏中间的
        if tip[3] != '-' and tip[1] != '-':
            p17_middle = (judge_point_distance['Serum Control'][1] + judge_point_distance['p24'][0]) // 2
        else:
            p17_middle = 820
        if abs((max_score_position[0][0] + max_score_position[0][1]) // 2 - p17_middle) <= \
                abs((max_score_position[1][0] + max_score_position[1][1]) // 2 - p17_middle):
            max_score_position = max_score_position[0]
        else:
            max_score_position = max_score_position[1]
    elif len(max_score_position) == 3:
        max_score_position = max_score_position[1]
    left, right = int(np.around(max_score_position[0])), int(np.around(max_score_position[1]))
    set_mp_tip(tip, 2, left, right, image_rotate_average_gray, image_average_gray, 'p55')
    judge_point_distance['p17'] = [left, right]


def set_mp_p120_p160_tip_and_distance(MpBand, black_position_result, tip, judge_point_distance,
                                      image_rotate_average_gray, image_average_gray):
    temp_left, temp_right = MpBand.SPEC_POINT_RANGE['gp120/gp160'][0], MpBand.SPEC_POINT_RANGE['gp120/gp160'][1]
    filter_by_area_cover = list(filter(lambda x: get_overlapping_area_coverage_rate(
        x[0], x[1], int(temp_left), int(temp_right)) > 0.6, black_position_result))
    if len(filter_by_area_cover) == 0:
        return
    # 根据分数排序，取分数最高的两个
    filter_by_area_cover = sorted(filter_by_area_cover, key=lambda x: x[2], reverse=True)[:2]
    if filter_by_area_cover[0][2] < 0.4:
        return
    # 根据x轴大小排序
    filter_by_area_cover = sorted(filter_by_area_cover, key=lambda x: x[0])
    if len(filter_by_area_cover) == 0:
        return
    elif len(filter_by_area_cover) == 1:
        left = int(np.around(filter_by_area_cover[0][0]))
        right = int(np.around(filter_by_area_cover[0][1]))
        if right - left > 130:
            set_mp_tip(tip, 10, left, int((right - left) * 0.4) + left, image_rotate_average_gray,
                       image_average_gray, 'gp120')
            set_mp_tip(tip, 11, int((right - left) * 0.4) + left, right, image_rotate_average_gray,
                       image_average_gray, 'gp160')
            judge_point_distance['gp120'] = [left, int((right - left) * 0.4) + left]
            judge_point_distance['gp160'] = [int((right - left) * 0.4) + left, right]
        else:
            if get_overlapping_area_coverage_rate(left, right, MpBand.POINT_RANGE['gp120'][0],
                                                  MpBand.POINT_RANGE['gp120'][1]) > \
                    get_overlapping_area_coverage_rate(left, right, MpBand.POINT_RANGE['gp160'][0],
                                                       MpBand.POINT_RANGE['gp160'][1]):
                set_mp_tip(tip, 10, left, right, image_rotate_average_gray, image_average_gray, 'gp120')
                judge_point_distance['gp120'] = [left, right]
            else:
                set_mp_tip(tip, 11, left, right, image_rotate_average_gray, image_average_gray, 'gp160')
                judge_point_distance['gp160'] = [left, right]
    else:
        set_mp_tip(tip, 10, int(np.around(filter_by_area_cover[0][0])), int(np.around(filter_by_area_cover[0][1])),
                   image_rotate_average_gray, image_average_gray, 'gp120')
        set_mp_tip(tip, 11, int(np.around(filter_by_area_cover[1][0])), int(np.around(filter_by_area_cover[1][1])),
                   image_rotate_average_gray, image_average_gray, 'gp160')
        judge_point_distance['gp120'] = [int(np.around(filter_by_area_cover[0][0])),
                                         int(np.around(filter_by_area_cover[0][1]))]
        judge_point_distance['gp160'] = [int(np.around(filter_by_area_cover[1][0])),
                                         int(np.around(filter_by_area_cover[1][1]))]
    # 如果gp160有，则单独用来识别gp120
    # if '-' == tip[10] and '+' == tip[11]:
    #     left_temp, right_temp = judge_point_distance['gp160'][0] - 100, judge_point_distance['gp160'][0] - 1
    #     min_value = np.min(image_rotate_average_gray[left_temp: right_temp])
    #     max_value = np.max(image_rotate_average_gray[left_temp: right_temp])
    #     if max_value - min_value < 15:
    #         return
    #     min_index = np.argmin(image_rotate_average_gray[left_temp: right_temp]) + left_temp
    #     # 向左找left，向右找right
    #     left = min_index - 1
    #     right = min_index + 1
    #     while left > left_temp and image_rotate_average_gray[left] < min_value + 15:
    #         left -= 1
    #     while right < right_temp and image_rotate_average_gray[right] < min_value + 15:
    #         right += 1
    #     set_mp_tip(tip, 10, left, right, image_rotate_average_gray, image_average_gray, 'gp120')
    #     judge_point_distance['gp120'] = [left, right]
    # for i in filter_by_area_cover:
    #     left = int(np.around(min(left, i[0])))
    #     right = int(np.around(max(right, i[1])))
    # if right - left > 130:
    #     set_mp_tip(tip, 10, left, int((right - left) * 0.4) + left, image_rotate_average_gray,
    #                image_average_gray, 'gp120')
    #     set_mp_tip(tip, 11, int((right - left) * 0.4) + left, right, image_rotate_average_gray,
    #                image_average_gray, 'gp160')
    #     judge_point_distance['gp120'] = [left, int((right - left) * 0.4) + left]
    #     judge_point_distance['gp160'] = [int((right - left) * 0.4) + left, right]
    # else:
    #     if get_overlapping_area_coverage_rate(left, right, MpBand.POINT_RANGE['gp120'][0],
    #                                           MpBand.POINT_RANGE['gp120'][1]) > \
    #             get_overlapping_area_coverage_rate(left, right, MpBand.POINT_RANGE['gp160'][0],
    #                                                MpBand.POINT_RANGE['gp160'][1]):
    #         set_mp_tip(tip, 10, left, right, image_rotate_average_gray, image_average_gray, 'gp120')
    #         judge_point_distance['gp120'] = [left, right]
    #     else:
    #         set_mp_tip(tip, 11, left, right, image_rotate_average_gray, image_average_gray, 'gp160')
    #         judge_point_distance['gp160'] = [left, right]


def set_mp_p51_p55_tip_and_distance(MpBand, black_position_result, tip, judge_point_distance,
                                    image_rotate_average_gray, image_average_gray, max_score_position):
    if max_score_position[0] is None:
        temp_left, temp_right = MpBand.SPEC_POINT_RANGE['p51/p55'][0], MpBand.SPEC_POINT_RANGE['p51/p55'][1]
        filter_by_area_cover = list(filter(lambda x: get_overlapping_area_coverage_rate(
            x[0], x[1], int(temp_left), int(temp_right)) > 0.6, black_position_result))
        if len(filter_by_area_cover) == 0:
            return
        max_score_position = sorted(filter_by_area_cover, key=lambda x: x[2], reverse=True)[:2]
        if max_score_position[0][2] < 0.4:
            return
        # max_score_position = sorted(filter_by_area_cover, key=lambda x: abs(middle - (x[0] + x[1]) / 2))[:2]
        max_score_position = sorted(max_score_position, key=lambda x: x[0])
    # 情况1 如果两个黑块重叠取分数大的黑块
    if len(max_score_position) > 1:
        if max_score_position[0][1] > max_score_position[1][1]:
            max_score_position = [max_score_position[0] if max_score_position[0][2] > max_score_position[1][2] else
                                  max_score_position[1]]
        elif max_score_position[0][1] > max_score_position[1][0]:
            middle = (max_score_position[0][1] + max_score_position[1][0]) // 2
            max_score_position[0] = [max_score_position[0][0], middle - 1]
            max_score_position[1] = [middle + 1, max_score_position[1][1]]
    if len(max_score_position) > 1:
        set_mp_tip(tip, 7, int(np.around(max_score_position[0][0])), int(np.around(max_score_position[0][1])),
                   image_rotate_average_gray, image_average_gray, 'p51')
        set_mp_tip(tip, 8, int(np.around(max_score_position[1][0])), int(np.around(max_score_position[1][1])),
                   image_rotate_average_gray, image_average_gray, 'p55')
        judge_point_distance['p51'] = [int(np.around(max_score_position[0][0])),
                                       int(np.around(max_score_position[0][1]))]
        judge_point_distance['p55'] = [int(np.around(max_score_position[1][0])),
                                       int(np.around(max_score_position[1][1]))]
    else:
        left, right = int(np.around(max_score_position[0][0])), int(np.around(max_score_position[0][1]))
        if right - left > 40:
            set_mp_tip(tip, 7, left, (left + right) // 2, image_rotate_average_gray, image_average_gray, 'p51')
            set_mp_tip(tip, 8, (left + right) // 2, right, image_rotate_average_gray, image_average_gray, 'p55')
            judge_point_distance['p51'] = [left, (left + right) // 2]
            judge_point_distance['p55'] = [(left + right) // 2, right]
        else:
            if tip[4] != '-' or tip[9] != '-':
                set_mp_tip(tip, 7, left, right, image_rotate_average_gray, image_average_gray, 'p51')
                judge_point_distance['p51'] = [left, right]
            if tip[2] != '-' or tip[3] != '-':
                if tip[7] != '-':
                    pass
                else:
                    set_mp_tip(tip, 8, left, right, image_rotate_average_gray, image_average_gray, 'p55')
                    judge_point_distance['p55'] = [left, right]
    # 判断情况2，如果p51有，p55没有，那么就判断p51往左边45个像素值，求其平均值是否小于右边45个像素值的最大值
    if tip[7] != '-' and tip[8] == '-':
        p51_right = judge_point_distance['p51'][1] + 5
        right_max = np.max(image_average_gray[p51_right: min(len(image_average_gray),
                                                             p51_right + 2 * MpBand.SPEC_POINT_RANGE['p51/p55'][2])])
        temp_p55_right = min(MpBand.SPEC_POINT_RANGE['p51/p55'][2], np.argmax(
            image_average_gray[p51_right: p51_right + 2 * MpBand.SPEC_POINT_RANGE['p51/p55'][2]])) + p51_right - 10
        temp_p55_right = p51_right if temp_p55_right < p51_right else temp_p55_right
        if temp_p55_right - p51_right > 10 and np.mean(image_average_gray[p51_right: temp_p55_right]) < right_max - 5:
            set_mp_tip(tip, 8, int(p51_right), int(temp_p55_right), image_rotate_average_gray, image_average_gray,
                       'p55')
            judge_point_distance['p55'] = [int(p51_right), int(temp_p55_right)]

        # p55_right =

JUDGMENT_POINT = ['Baseline', 'gp160', 'gp120', 'p66', 'p51/p55', 'gp41', 'p31', 'p24', 'p17', 'Control', 'HIV-2']
POINT_NUMS = 11
POINT_RANGE = {
    'Baseline': [0, 20, 15],
    'gp160': [20, 160, 25],  # 这个区域中最暗的
    'gp120': [100, 280, 70],  # 从右往左最暗的
    'p66': [320, 445, 40],
    'p51/p55': [450, 550, 35],
    'gp41': [580, 660, 50],
    'p31': [780, 890, 25],
    'p24': [990, 1100, 25],
    'p17': [1200, 1360, 30],
    'Control': [1400, 1580, 25],
    'HIV-2': [1580, 1800, 30]
}

def get_imt_tip_and_distance(digital_position_result, black_position_result, image_rotate_average_gray):
    """
    从baseline开始
    :return:
    """
    # baseline 位置
    i = get_imt_baseline_begin_position(digital_position_result,
                                              black_position_result)
    if i == 0:
        # is_success = False
        # error_msg += "未找到Baseline"
        return [], {}
    truth_baseline_start = i
    imt_judgement_point_range = POINT_RANGE
    standard_baseline_start = imt_judgement_point_range['Baseline'][0]
    tip = ['-'] * len(imt_judgement_point_range)
    judge_point_distance = {}
    for j, entry in enumerate(imt_judgement_point_range.items()):
        temp_left = int(entry[1][0]) - standard_baseline_start + truth_baseline_start
        temp_right = int(entry[1][1]) - standard_baseline_start + truth_baseline_start
        judge_point_distance[entry[0]] = [temp_left, temp_right]
        middle = (int(entry[1][0]) + int(entry[1][1])) / 2
        # 先过滤掉面积覆盖率<60%的
        filter_by_area_cover = list(filter(lambda x:
                                           get_overlapping_area_coverage_rate(x[0] - truth_baseline_start,
                                                                                    x[1] -
                                                                                    truth_baseline_start,
                                                                                    int(entry[1][0]),
                                                                                    int(entry[1][1])) > 0.6 and x[
                                               2] > 0.3,
                                           black_position_result))
        # 再找距离标记范围中心点最近的黑块
        if len(filter_by_area_cover) == 0:
            continue
        # gp160特殊点，如果再gp160这个范围内找到多个黑块，使用该黑块范围内灰度值最小的那个
        if 'gp160' == entry[0] or 'p66' == entry[0]:
            max_score_position = sorted(filter_by_area_cover,
                                        key=lambda x: np.min(image_rotate_average_gray[int(np.around(x[0])):
                                                                                            int(np.around(x[1]))]))[
                0]
        elif 'Control' == entry[0] or 'HIV-2' == entry[0]:
            filter_by_area_cover = list(filter(lambda x: (x[1] - x[0] > 10) and x[2] > 0.7, filter_by_area_cover))
            if len(filter_by_area_cover) == 0:
                continue
            max_score_position = sorted(filter_by_area_cover,
                                        key=lambda x: np.min(image_rotate_average_gray[int(np.around(x[0])):
                                                                                            int(np.around(x[1]))]))[
                0]
        elif 'p17' == entry[0] or 'p51/p55' == entry[0]:
            max_score_position = sorted(filter_by_area_cover, key=lambda x: x[0], reverse=True)[0]
        elif 'p31' == entry[0]:
            # 左边
            max_score_position = sorted(filter_by_area_cover, key=lambda x: x[0])[0]
        else:
            max_score_position = sorted(filter_by_area_cover,
                                        key=lambda x: abs(middle - (x[0] + x[1] - 2 * truth_baseline_start) / 2))[0]
        # print(entry[0], max_score_position)
        left, right = int(np.around(max_score_position[0])), int(np.around(max_score_position[1]))
        set_imt_tip(tip, j, left, right, image_rotate_average_gray)
        judge_point_distance[entry[0]] = [left, right]
    return tip, judge_point_distance
