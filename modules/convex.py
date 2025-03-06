import os
import io
import numpy
import pickle
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
from utils.directoryHandler import check_processed_data, create_fig_dir
from utils.logger import log_execution_time, log_progress, log

def apply_windowing(pixels, window_center, window_width):
    """
    Windowing을 적용하는 함수

    Args:
        pixels (numpy.ndarray): 픽셀 데이터
        window_center (int): 윈도윙 중심값
        window_width (int): 윈도윙 폭

    Returns:    
        numpy.ndarray: Windowing
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    return numpy.clip(pixels, img_min, img_max)

def get_adjusted_pixel_with_window(dicom_file_path, window_center, window_width):
    """
    Windowing을 적용한 픽셀을 반환하는 함수

    Args:
        dicom_file_path (str): DICOM 파일 경로
        window_center (int): 윈도윙 중심값
        window_width (int): 윈도윙 폭

    Returns:
        numpy.ndarray: Windowing
    """
    dicom_dataset = pydicom.dcmread(dicom_file_path, force=True)
    pixels = dicom_dataset.pixel_array
    adjusted_pixels = apply_windowing(pixels, window_center, window_width)
    return adjusted_pixels

def get_body_mask(dicom_file_path):
    """
    Body Mask를 반환하는 함수

    Args:
        dicom_file_path (str): DICOM 파일 경로

    Returns:
        numpy.ndarray: Body Mask
    """
    pixels = get_adjusted_pixel_with_window(dicom_file_path, 20, 1000)
    mask = (pixels > 0).astype(numpy.uint8)
    return mask

def get_convex_hull(slice_image):
    """
    Convex Hull을 반환하는 함수

    Args:
        slice_image (numpy.ndarray): Slice 이미지

    Returns:
        tuple: points, convex_hull
    """
    points = numpy.column_stack(numpy.where(slice_image > 0))
    if len(points) < 3:
        return None, None
    try:
        hull = ConvexHull(points)
        convex_hull = points[hull.vertices][:, ::-1]
        convex_hull[:, 1] = slice_image.shape[0] - convex_hull[:, 1]
        return points, convex_hull
    except:
        return None, None

def adjust_convex_hull(convex_hull, binary_mask, move_down_units=4):
    """
    Convex Hull을 조정하는 함수

    Args:
        convex_hull (numpy.ndarray): Convex Hull
        binary_mask (numpy.ndarray): Binary Mask
        move_down_units (int): 이동 단위

    Returns:
        numpy.ndarray: 조정된 Convex Hull
    """
    lung_bottom = numpy.max(numpy.where(binary_mask > 0)[0])

    hull_bottom_index = numpy.argmax(convex_hull[:, 1])
    hull_bottom_y = convex_hull[hull_bottom_index, 1]

    if hull_bottom_y < lung_bottom:
        new_convex_hull = convex_hull.copy()
        new_convex_hull[:, 1] += move_down_units
        return new_convex_hull
    else:
        return convex_hull

@log_execution_time
def convex(working_dir: str, body_mask_path: str, common_data: dict):
    """
    Convex Hull을 생성하는 함수

    Args:
        working_dir (str): 작업 디렉터리 경로(./data/working/{환자 ID}/{in, ex})
        body_mask_path (str): Body Mask 경로
        common_data (dict): Convex와 Masking에 필요한 데이터

    Returns:
        list: Convex Hull Plot 데이터
    """   
    plot_data = []
    save_file = "plot_convex.pkl"
    save_path = os.path.join(working_dir, save_file)
    fig_dir = create_fig_dir(working_dir, "convex")

    if check_processed_data(save_path):
        log("info", f"처리된 데이터가 존재합니다. 해당 데이터를 불러옵니다. ({save_path})")
        processed_data = pickle.load(open(save_path, 'rb'))
        plot_data = processed_data
    else:
        masked_pixel_data = common_data['masked_pixel_data'] # 512 x 512 x n

        body_mask = get_body_mask(body_mask_path) # n x 512 x 512
        num_slices = body_mask.shape[0] 

        adjusted_convex_hull_list = []
        
        for slice_index in range(min(num_slices, masked_pixel_data.shape[2])):
            log_progress(slice_index + 1, num_slices, "STEP 1: Convex Hull 생성 중...")
            slice_image = body_mask[slice_index] # 512 x 512
            points, convex_hull = get_convex_hull(slice_image)
            if convex_hull is not None and len(convex_hull) >= 3:
                adjusted_convex_hull = adjust_convex_hull(convex_hull, slice_image)
                adjusted_convex_hull_list.append(adjusted_convex_hull)
            else:
                adjusted_convex_hull_list.append(None)
        log("success", "STEP 1: Convex Hull 생성이 완료되었습니다.")
        
        for slice_index in range(min(num_slices, masked_pixel_data.shape[2])):
            log_progress(slice_index + 1, num_slices, "STEP 2: Convex Hull 시각화 중...")
            masked_image = masked_pixel_data[:, :, slice_index]
            convex_hull = adjusted_convex_hull_list[slice_index]

            fig, ax = plt.subplots(figsize=(masked_image.shape[1] / 100, masked_image.shape[0] / 100), dpi=100)
            if convex_hull is not None:
                convex_hull[:, 1] = masked_image.shape[0] - convex_hull[:, 1]
                polygon_patch = patches.Polygon(convex_hull, closed=True, edgecolor='black', facecolor='black', fill=True)
                ax.add_patch(polygon_patch)

            ax.imshow(masked_image, cmap='gray')
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(os.path.join(fig_dir, f"{slice_index:03d}.png"))
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plot_data.append(buf.read())
            plt.close(fig)
        log("success", "STEP 2: Convex Hull 시각화가 완료되었습니다.")
    
        pickle.dump(plot_data, open(save_path, 'wb'))
        log("success", f"Convex Hull Plot 데이터를 저장하였습니다. ({save_path})")

    return plot_data