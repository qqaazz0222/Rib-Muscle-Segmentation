import numpy
import cv2
import numpy
import matplotlib.pyplot as plt
from utils.logger import log, log_execution_time, log_progress
from modules.extractor import save

@log_execution_time
def post_processing(extracted_muscle_image_dict: dict, metadata_dict: dict):
    abs_point_dict = calculate_linear_abs_point(metadata_dict)
    post_processed_muscle_image_dict = {"in": {}, "ex": {}, "in_abs": {}, "ex_abs": {}}
    for category in ["in", "ex"]:
        sn_list = extracted_muscle_image_dict[category].keys()
        for sn in sn_list:
            log("info", "Processing Series Number: " + sn)
            extracted_muscle_image_list = extracted_muscle_image_dict[category][sn]
            processed_image_list = []
            abs_masked_image_list = []
            for idx in range(len(extracted_muscle_image_list)):
                log_progress(idx + 1, len(extracted_muscle_image_list), f"Post-Processing {'Inhalation' if category == 'in' else 'Exhalation'} Muscle Images")
                flag = [False if idx == 0 else True, False if idx == len(extracted_muscle_image_list) - 1 else True]
                cur_image = extracted_muscle_image_list[idx]
                if flag[0]:
                    prev_image = extracted_muscle_image_list[idx - 1]
                    prev_processed_image = processed_image_list[-1]
                else:
                    prev_image = cur_image
                    prev_processed_image = cur_image
                if flag[1]:
                    next_image = extracted_muscle_image_list[idx + 1]
                else:
                    next_image = cur_image
                prev_image_mask = (prev_image[:, :, 0] == 255) & (prev_image[:, :, 1] == 0) & (prev_image[:, :, 2] == 0)
                prev_image_reverse_mask = (prev_image[:, :, 0] == 0) & (prev_image[:, :, 1] == 0) & (prev_image[:, :, 2] == 0)
                prev_processed_image_mask = (prev_processed_image[:, :, 0] == 255) & (prev_processed_image[:, :, 1] == 0) & (prev_processed_image[:, :, 2] == 0) 
                prev_processed_image_reverse_mask = (prev_processed_image[:, :, 0] == 0) & (prev_processed_image[:, :, 1] == 0) & (prev_processed_image[:, :, 2] == 0)
                next_image_mask = (next_image[:, :, 0] == 255) & (next_image[:, :, 1] == 0) & (next_image[:, :, 2] == 0)
                next_image_reverse_mask = (next_image[:, :, 0] == 0) & (next_image[:, :, 1] == 0) & (next_image[:, :, 2] == 0)
                union_image = (prev_image_mask.astype(numpy.float32) / 3 +
                            prev_processed_image_mask.astype(numpy.float32) / 3 +
                            next_image_mask.astype(numpy.float32) / 3)
                union_reverse_image = (prev_image_reverse_mask.astype(numpy.float32) / 3 +
                                    prev_processed_image_reverse_mask.astype(numpy.float32) / 3 +
                                        next_image_reverse_mask.astype(numpy.float32) / 3)
                black_pixel_regions = (prev_image_reverse_mask & next_image_reverse_mask & prev_processed_image_reverse_mask)

                processed_image = cur_image.copy()
                processed_image[union_image > 0.7] = [255, 0, 0]
                processed_image[union_reverse_image > 0.9] = [0, 0, 0]
                processed_image[black_pixel_regions] = [0, 0, 0]
                try:
                    processed_image, abs_masked_image = masking_abs(
                        processed_image, 
                        metadata_dict[category][sn][idx]["abs"]["base_point"], 
                        abs_point_dict[category][sn]["left"][idx], 
                        abs_point_dict[category][sn]["right"][idx])
                except:
                    abs_masked_image = processed_image.copy()
                processed_image_list.append(processed_image)
                abs_masked_image_list.append(abs_masked_image)
                
                # if category == "ex" and idx == 168:
                #     # metadata_dict[category][idx]["abs"]["base_point"]
                #     # abs_point_dict[category]["left"][idx] 
                #     # abs_point_dict[category]["right"][idx]
                #     print("base_point", metadata_dict[category][idx]["abs"]["base_point"])
                #     print("left_point", abs_point_dict[category]["left"][idx])
                #     print("right_point", abs_point_dict[category]["right"][idx])
                #     save(cur_image, "cur_image.png")
                #     save(union_image, "union_image.png")
                #     save(union_reverse_image, "union_reverse_image.png")
                #     save(processed_image, "processed_image.png")
                #     save(abs_masked_image, "abs_masked_image.png")
                    

            post_processed_muscle_image_dict[category][sn] = processed_image_list
            post_processed_muscle_image_dict[f"{category}_abs"][sn] = abs_masked_image_list
    return post_processed_muscle_image_dict

def interpolate_coords(input_coords: list):
    for i in range(len(input_coords)):
        if 176 <= input_coords[i][0] <= 336 and input_coords[i][1] >= 255:
            input_coords[i] = tuple([0, 0])
    coords = numpy.array(input_coords, dtype=float)
    interpolated_coords = numpy.zeros_like(coords)
    valid_coords = numpy.array([coord for coord in coords if coord[0] not in [0, -1]], dtype=float)
    
    if len(valid_coords) > 0:
        x = valid_coords[:, 0]
        y = valid_coords[:, 1]
        
        coefficients = numpy.polyfit(y, x, 4)
        polynomial = numpy.poly1d(coefficients)
        x_fitted = polynomial(y)
        valid_coords = numpy.column_stack((x_fitted, y))
    
    valid_idx = 0
    for i in range(len(coords)):
        if coords[i][0] not in [0, -1]:
            interpolated_coords[i] = valid_coords[valid_idx]
            valid_idx += 1
        else:
            interpolated_coords[i] = coords[i]
    
    start_idx = next((i for i, coord in enumerate(interpolated_coords) if coord[1] not in [0, -1]), None)
    end_idx = next((i for i, coord in enumerate(reversed(interpolated_coords)) if coord[1] not in [0, -1]), None)
    if end_idx is not None:
        end_idx = len(interpolated_coords) - 1 - end_idx
        
    if start_idx is None or end_idx is None:
        return []

    roi = interpolated_coords[start_idx:end_idx + 1]
    for i in range(len(roi)):
        if numpy.array_equal(roi[i], [-1, -1]):
            neighbors = []
            if i > 0 and not numpy.array_equal(roi[i - 1], [-1, -1]):
                neighbors.append(roi[i - 1])
            if i < len(roi) - 1 and not numpy.array_equal(roi[i + 1], [-1, -1]):
                neighbors.append(roi[i + 1])
            if neighbors:
                roi[i] = numpy.mean(neighbors, axis=0)
    roi = roi[numpy.argsort(roi[:, 1])]
    
    coords = interpolated_coords
    coords[start_idx:end_idx + 1] = roi
    
    last_idx = 0
    last_coord = None
    for i in range(len(coords) - 1, -1, -1):
        check_num = int(coords[i][1])
        if check_num == 0 or check_num == -1:
            continue
        else:
            last_idx = i
            last_coord = coords[i]
            break
    for i in range(last_idx + 1, len(coords)):
        coords[i] = last_coord
    
    return coords

def calculate_linear_abs_point(metadata_dict: dict):
    abs_point_dict = {"in": {}, "ex": {}}
    for category in ["in", "ex"]:
        sn_list = metadata_dict[category].keys()
        for sn in sn_list:
            metadata_list = metadata_dict[category][sn]
            left_point_list = []
            right_point_list = []
            for metadata in metadata_list:
                idx = metadata["idx"]
                abs_data = metadata["abs"]
                left_point = abs_data["left_point"]
                right_point = abs_data["right_point"]
                left_point_list.append(left_point)
                right_point_list.append(right_point)
            interpolate_left_coords = interpolate_coords(left_point_list)
            interpolate_right_coords = interpolate_coords(right_point_list)
            abs_point_dict[category][sn] = {
                "left": interpolate_left_coords,
                "right": interpolate_right_coords
            }
    return abs_point_dict
    
def masking_abs(pixel_array: numpy.ndarray, center: tuple, left_point: tuple, right_point: tuple):
    vis_pixel_array = pixel_array.copy()
    h, w = pixel_array.shape[:2]
    center_point = (center[0], center[1])
    left_point = (int(left_point[0]), int(left_point[1]))
    right_point = (int(right_point[0]), int(right_point[1]))
    
    if center_point[0] != left_point[0]:
        slope = (center_point[1] - left_point[1]) / (center_point[0] - left_point[0])
        left_y = int(left_point[1] - slope * left_point[0])
    else:
        left_y = left_point[1]
        
    if center_point[0] != right_point[0]:
        slope = (center_point[1] - right_point[1]) / (center_point[0] - right_point[0])
        right_y = int(center_point[1] + slope * (w - center_point[0]))
    else:
        right_y = right_point[1]
    
    left_top_point = (0, 0)
    left_bottom_point = (0, left_y)
    right_top_point = (w, 0)
    right_bottom_point = (w, right_y)
    points = numpy.array([
        left_top_point,
        left_bottom_point,
        center_point,
        right_bottom_point,
        right_top_point
    ], dtype=numpy.int32)
    
    flag = left_y == 0 or right_y == 0
    if not flag:
        cv2.drawContours(pixel_array, [points], -1, (0, 0, 0), -1)
        cv2.drawContours(vis_pixel_array, [points], -1, (0, 255, 255), 2)
        cv2.circle(vis_pixel_array, center_point, 5, (0, 255, 0), -1) # 중심점 시각화
    # cv2.drawContours(pixel_array, [points], -1, (0, 255, 0), 2) # 복근 마스킹 영역 시각화
    return pixel_array, vis_pixel_array