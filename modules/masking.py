import os
import numpy
import pickle
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from utils.directoryHandler import check_processed_data, create_fig_dir
from utils.logger import log_execution_time, log_progress, log

def draw_contour(image, contour):
    """
    컨투어를 그리는 함수
    
    Args:
        image (numpy.ndarray): 이미지 데이터
        contour (numpy.ndarray): 컨투어 데이터

    Returns:
        numpy.ndarray: 컨투어가 그려진 이미지 데이터
    """
    image_copy = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    cv.drawContours(image_copy, contour, -1, (0, 0, 255), 2)
    return image_copy

def remove_noise(image):
    """
    노이즈를 제거하는 함수
    
    Args:
        image (numpy.ndarray): 이미지 데이터

    Returns:
        numpy.ndarray: 노이즈가 제거된 이미지 데이터
        """
    kernel = numpy.ones((7, 7), numpy.uint8)
    closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    
    contours, _ = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    min_contour_area = 200
    filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]
    
    return filtered_contours

def move_contour_inside(contour, move_distance_x, move_distance_y):
    """
    컨투어를 상하 또는 좌우로 각각 이동시키는 함수

    Args:
        contour (numpy.ndarray): 컨투어 데이터
        move_distance_x (int): X축 이동 거리
        move_distance_y (int): Y축 이동 거리

    Returns:
        numpy.ndarray: 이동된 컨투어 데이터
    """
    center_x = numpy.mean(contour[:, 0, 0])
    center_y = numpy.mean(contour[:, 0, 1])
    
    new_contour = []
    for point in contour:
        x, y = point[0]
        distance_to_center_x = x - center_x
        distance_to_center_y = y - center_y
        
        if distance_to_center_x != 0 or distance_to_center_y != 0:
            move_vector_x = (distance_to_center_x / numpy.sqrt(distance_to_center_x**2 + distance_to_center_y**2)) * move_distance_x
            move_vector_y = (distance_to_center_y / numpy.sqrt(distance_to_center_x**2 + distance_to_center_y**2)) * move_distance_y
            
            new_contour.append([[x - move_vector_x, y - move_vector_y]])
        else:
            new_contour.append([[x, y]])
    
    new_contour = numpy.array(new_contour, dtype=numpy.int32)
    return new_contour

def draw_contour_on_blank(image_shape, contours, fill=False):
    """
    빈 이미지에 컨투어를 그리는 함수

    Args:
        image_shape (tuple): 이미지 크기
        contours (numpy.ndarray): 컨투어 데이터
        fill (bool): 채우기 여부

    Returns:
        numpy.ndarray: 컨투어가 그려진 이미지 데이터
    """
    blank_image = numpy.zeros((image_shape[0], image_shape[1], 3), dtype=numpy.uint8)
    if fill:
        cv.drawContours(blank_image, contours, -1, (0, 0, 255), thickness=cv.FILLED)
    else:
        cv.drawContours(blank_image, contours, -1, (0, 0, 255), thickness=2)
    return blank_image

def interpolate_line(pt1, pt2):
    """
    두 점을 연결하는 선을 보간하는 함수

    Args:
        pt1 (tuple): 첫 번째 점
        pt2 (tuple): 두 번째 점

    Returns:
        list: 보간된 점들
    """
    x1, y1 = pt1
    x2, y2 = pt2
    line_points = []
    num_points = int(numpy.hypot(x2 - x1, y2 - y1))
    for i in range(num_points + 1):
        t = i / num_points
        x = int((1 - t) * x1 + t * x2)
        y = int((1 - t) * y1 + t * y2)
        line_points.append((x, y))
    return line_points

def remove_noise_limit(image):
    """
    노이즈를 제거하는 함수

    Args:
        image (numpy.ndarray): 이미지 데이터

    Returns:
        numpy.ndarray: 노이즈가 제거된 이미지 데이터
    """
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv.contourArea(contour) > 30]
    return filtered_contours

def adjust_contour_to_red_line(contour, red_line_points):
    """
    컨투어를 빨간색 선에 맞게 조정하는 함수

    Args:
        contour (numpy.ndarray): 컨투어 데이터
        red_line_points (list): 빨간색 선의 점들

    Returns:
        numpy.ndarray: 조정된 컨투어 데이터
    """
    adjusted_contour = []
    for point in contour:
        x, y = point[0]
        if is_above_red_line(x, y, red_line_points):
            new_y = interpolate_red_line_y(x, red_line_points)
            adjusted_contour.append([[x, new_y]])
        else:
            adjusted_contour.append([[x, y]])
    return numpy.array(adjusted_contour, dtype=numpy.int32)

def is_above_red_line(x, y, red_line_points):
    """
    점이 빨간색 선 위에 있는지 확인하는 함수

    Args:
        x (int): X 좌표
        y (int): Y 좌표
        red_line_points (list): 빨간색 선의 점들

    Returns:
        bool: 빨간색 선 위에 있으면 True, 아니면 False
    """
    for i in range(len(red_line_points) - 1):
        x1, y1 = red_line_points[i]
        x2, y2 = red_line_points[i + 1]
        if x1 <= x <= x2 or x2 <= x <= x1:
            t = (x - x1) / (x2 - x1) if x1 != x2 else 0
            y_on_line = y1 + t * (y2 - y1)
            return y > y_on_line
    return False

def interpolate_red_line_y(x, red_line_points):
    """
    빨간색 선 위의 점을 보간하는 함수

    Args:
        x (int): X 좌표
        red_line_points (list): 빨간색 선의 점들

    Returns:
        int
    """
    for i in range(len(red_line_points) - 1):
        x1, y1 = red_line_points[i]
        x2, y2 = red_line_points[i + 1]
        if x1 <= x <= x2 or x2 <= x <= x1:
            t = (x - x1) / (x2 - x1) if x1 != x2 else 0
            return y1 + t * (y2 - y1)
    return 0

def create_figure(slice_img, contour_img, moved_contours, red_line_points, i):
    """
    그래프를 생성하는 함수
    
    Args:
        slice_img (numpy.ndarray): 슬라이스 이미지
        contour_img (numpy.ndarray): 컨투어 이미지
        moved_contours (list): 이동된 컨투어 리스트
        red_line_points (list): 빨간색 선의 점들
        i (int): 인덱스

    Returns:
        matplotlib.figure.Figure: 그래프
    """
    fig, ax = plt.subplots(figsize=(slice_img.shape[1] / 100, slice_img.shape[0] / 100), dpi=100)
    if slice_img.dtype == numpy.uint16:
        slice_img_normalized = cv.normalize(slice_img, None, 0, 255, cv.NORM_MINMAX)
        slice_img_normalized = numpy.uint8(slice_img_normalized)
    else:
        slice_img_normalized = slice_img
    overlay_img = numpy.zeros((slice_img_normalized.shape[0], slice_img_normalized.shape[1], 3), dtype=numpy.uint8)
    overlay_img[:, :, 0] = slice_img_normalized
    overlay_img[:, :, 1] = slice_img_normalized
    overlay_img[:, :, 2] = slice_img_normalized
    if len(contour_img.shape) == 2:
        contour_img_color = cv.cvtColor(contour_img, cv.COLOR_GRAY2BGR)
    elif len(contour_img.shape) == 3 and contour_img.shape[2] == 3:
        contour_img_color = contour_img
    else:
        raise ValueError("Invalid number of channels in contour_img.")
    overlay_img = cv.addWeighted(overlay_img, 1, contour_img_color, 1, 0)

    ax.imshow(overlay_img)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for contour in moved_contours:
        adjusted_contour = adjust_contour_to_red_line(contour, red_line_points)
        ax.plot(adjusted_contour[:, 0, 0], adjusted_contour[:, 0, 1], 'k', linewidth=2)
        ax.fill(adjusted_contour[:, 0, 0], adjusted_contour[:, 0, 1], color='k', alpha=1)
    if red_line_points is not None:
        red_line_x = numpy.array([pt[0] for pt in red_line_points])
        red_line_y = numpy.array([pt[1] for pt in red_line_points])
        ax.plot(red_line_x, red_line_y, color='k', linewidth=2)
    plt.close(fig)
    return fig

@log_execution_time
def masking(working_dir: str, common_data: dict):
    """
    마스킹을 수행하는 함수

    Args:
        working_dir (str): 작업 디렉터리 경로(./data/working/{환자 ID}/{in, ex})
    """
    plot_data = []
    save_file = "plot_masking.pkl"
    save_path = os.path.join(working_dir, save_file)
    fig_dir = create_fig_dir(working_dir, "masking")

    if check_processed_data(save_path):
        log("info", f"처리된 데이터가 존재합니다. 해당 데이터를 불러옵니다. ({save_path})")
        processed_data = pickle.load(open(save_path, 'rb'))
        plot_data = processed_data
    else:
        masked_pixel_data = common_data['masked_pixel_data'] # 512 x 512 x n
        

        move_distance_y = 40
        move_distance_x = 25
        filled_coordinates = []
        coordinates_list = []
        hu_min = 300
        hu_max = 1900
        hu_range_mask = numpy.logical_and(masked_pixel_data >= hu_min, masked_pixel_data <= hu_max)
        masked_hu_pixel_array_list = masked_pixel_data * hu_range_mask
        num_slices = masked_hu_pixel_array_list.shape[2]
        filter_size = 3
        red_line_points_list = []

        for i in range(masked_pixel_data.shape[2]):
            log_progress(i + 1, masked_pixel_data.shape[2], "STEP 1: 노이즈 제거 중...")
            slice_img = masked_pixel_data[:, :, i].astype(numpy.uint8) # 512 x 512
            filtered_contours = remove_noise(slice_img)
            draw_contour(slice_img, filtered_contours)
        log('success', 'Step 1: 노이즈 제거 완료')

        for i in range(masked_pixel_data.shape[2]):
            log_progress(i + 1, masked_pixel_data.shape[2], "STEP 2: 컨투어 이동 중...")
            slice_img = masked_pixel_data[:, :, i].astype(numpy.uint8)
            filtered_contours = remove_noise(slice_img)
            moved_contours = [move_contour_inside(contour, move_distance_x, move_distance_y) for contour in filtered_contours]
            contour_img = draw_contour_on_blank(slice_img.shape, moved_contours, fill=True)
            filled_coords = numpy.column_stack(numpy.where(contour_img[:, :, 2] == 255))
            filled_coordinates.extend(filled_coords)
        log('success', 'Step 2: 컨투어 이동 완료')

        for i in range(num_slices):
            log_progress(i + 1, num_slices, "STEP 3: HU 범위 적용 중...")
            mid_x = masked_hu_pixel_array_list[:, :, i].shape[1] // 2
            mid_y = masked_hu_pixel_array_list[:, :, i].shape[0] // 2
            denoised_slice = median_filter(masked_hu_pixel_array_list[:, :, i], size=filter_size)
            first_y = None
            for y in range(mid_y, masked_hu_pixel_array_list.shape[0]):
                if hu_min <= denoised_slice[y, mid_x] <= hu_max:
                    first_y = y
                    break
            coordinates_list.append((mid_x, first_y))
        log('success', 'Step 3: HU 범위 적용 완료')

        bone_regions = masked_hu_pixel_array_list

        for idx in range(bone_regions.shape[2]):
            log_progress(idx + 1, bone_regions.shape[2], "STEP 4: 임계선 추출 중...")
            slice_image = bone_regions[:, :, idx].astype(numpy.uint8)  # uint8로 변환
            y_non_zero_pixels = numpy.nonzero(slice_image)[0]
            if len(y_non_zero_pixels) == 0:
                continue
            y_min = numpy.min(y_non_zero_pixels)
            y_max = numpy.max(y_non_zero_pixels)
            y_average = (y_max + y_min) / 2
            filtered_contours = remove_noise_limit(slice_image)
            contour_centers = []
            for contour in filtered_contours:
                M = cv.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    contour_centers.append((cx, cy))
            mid_x, _ = coordinates_list[idx]
            filtered_centers = [center for center in contour_centers if abs(center[0] - mid_x) > 20]
            filtered_centers = [center for center in filtered_centers if center[1] >= y_average]
            valid_coordinates = [coordinates_list[idx]] if coordinates_list[idx][0] is not None and coordinates_list[idx][1] is not None else []
            combined_list = sorted(filtered_centers + valid_coordinates, key=lambda x: x[0])
            red_line_points_list.append(combined_list)
        log('success', 'Step 4: 임계선 추출 완료')

        for i in range(masked_pixel_data.shape[2]):
            log_progress(i + 1, masked_pixel_data.shape[2], "STEP 5: 그래프 생성 중...")
            slice_img = masked_pixel_data[:, :, i].astype(numpy.uint16)
            filtered_contours = remove_noise(slice_img.astype(numpy.uint8))  # uint8로 변환
            moved_contours = [move_contour_inside(contour, move_distance_x, move_distance_y) for contour in filtered_contours]
            contour_img = draw_contour_on_blank(slice_img.shape, moved_contours)
            red_line_points = red_line_points_list[i]

            fig = create_figure(slice_img, contour_img, moved_contours, red_line_points, i)
            fig.savefig(os.path.join(fig_dir, f"{i:03d}.png"))
            fig.set_size_inches(5.12, 5.12)
            fig.set_dpi(100)
            plot_data.append(fig)
        log('success', 'Step 5: 그래프 생성 완료')

        pickle.dump(plot_data, open(save_path, 'wb'))
        log("success", f"Masking Plot 데이터를 저장하였습니다. ({save_path})")

    return plot_data