import os
import torch
import pickle
from argparse import ArgumentParser
from model.model import init_model
from modules.classifier import classify
from modules.preprocess import preprocess
from modules.convex import convex
from modules.masking import masking
from utils.directoryHandler import check_processed_data
from utils.dicomHandler import *
from utils.imageHandler import *
from utils.logger import *

args = None

@log_execution_time
def init_args():
    """
    인자 초기화 함수

    Returns:
        args (ArgumentParser): 인자
    """
    parser = ArgumentParser()
    parser.add_argument('--patient_id', type=str, default="10149710", help='환자 아이디(단일 환자만 분석)')
    parser.add_argument('--input_dir', type=str, default='data/input', help='입력(원본데이터) 디렉토리')
    parser.add_argument('--output_dir', type=str, default='data/output', help='출력 디렉토리')
    parser.add_argument('--working_dir', type=str, default='data/working', help='작업 디렉토리')
    parser.add_argument('--label_dir', type=str, default='data/label', help='라벨(정답데이터) 디렉토리')
    parser.add_argument('--weight', type=str, default='model/checkpoint.pth', help='모델 가중치 파일 경로')
    return parser.parse_args()

@log_execution_time
def load_common_data(working_dir: str):
    """
    Convex와 Masking에 필요한 데이터를 로드하는 함수

    Args:
        working_dir (str): 작업 디렉터리 경로

    Returns:
        tuple: flag, Convex와 Masking에 필요한 데이터
    """ 
    if check_processed_data(os.path.join(working_dir, 'common_data.pkl')):
        common_data = pickle.load(open(os.path.join(working_dir, 'common_data.pkl'), 'rb'))
        return True, common_data
    common_data = {
        'dicom_data': None,
        'pixel_data': None,
        'body_mask_data': None,
        'masked_pixel_data': None
    }
    log_progress(1, 4, "Load DICOM Data...")
    dicom_data = get_dicom_data_list(working_dir)
    if len(dicom_data) == 0:
        return False, common_data
    log_progress(2, 4, "Load Series Data...")
    pixel_data = get_dicom_series(dicom_data, True)
    log_progress(3, 4, "Load Body Mask Data(This task may take 3 minutes or more)...")
    body_mask_data = get_body_mask_series_with_DBSCAN(dicom_data)
    log_progress(4, 4, "Load Masked Pixel Data...")
    masked_pixel_data = pixel_data * body_mask_data
    common_data['dicom_data'] = dicom_data
    common_data['pixel_data'] = pixel_data
    common_data['body_mask_data'] = body_mask_data
    common_data['masked_pixel_data'] = masked_pixel_data
    pickle.dump(common_data, open(os.path.join(working_dir, 'common_data.pkl'), 'wb'))
    return True, common_data

@log_execution_time
def predict(working_dir: str):
    """
    CNN 모델을 통해 평가 방법을 정의하는 함수

    Args:
        working_dir (str): 작업 디렉터리 경로

    Returns:
        tuple: 예측된 클래스 리스트(predicted_class_zip), 확률 리스트(probabilities)
    """
    model = init_model(args.weight)
    predicted_class_zip = []
    probabilities = []
    
    dicom_file_list = get_dicom_files(working_dir)

    for file_path in dicom_file_list:
        log_progress(dicom_file_list.index(file_path) + 1, len(dicom_file_list), f"Predicting({file_path})...")
        image = get_preprocessed_dicom_image(file_path)

        with torch.no_grad():
            output = model(image)
            probability = output.item()
            predicted_class = 1 if probability > 0.5 else 0

        predicted_class_zip.append(predicted_class)
        probabilities.append(probability)

    return predicted_class_zip

@log_execution_time
def filtering(predicted_class_zip: list, convex_list: list, masking_list: list):
    """
    퍙기 빙밥에 따라 Convex와 Masking 정보를 필터링하는 함수

    Args:
        predicted_class_zip (list): 예측된 클래스 리스트
        convex_list (list): Convex 정보 리스트
        masking_list (list): Masking 정보 리스트

    Returns:
        tuple: 필터링된 Convex 정보 리스트(filtered_convex_info), 필터링된 Masking 정보 리스트(filtered_masking_info)
    """
    filtered_convex_info = []
    filtered_masking_info = []

    for idx, predicted_class in enumerate(predicted_class_zip):
        if predicted_class == 0:
            if idx < len(convex_list):
                convex_info = convex_list[idx]
                filtered_convex_info.append(convex_info)
        elif predicted_class == 1:
            if idx < len(masking_list):
                masking_info = masking_list[idx]
                filtered_masking_info.append(masking_info)

    return filtered_convex_info, filtered_masking_info

def main(args: ArgumentParser):
    """
    메인 함수

    Args:
        args (ArgumentParser): 인자

    Returns:
        None
    """
    working_dir = classify(args.patient_id, args.input_dir, args.working_dir)
    metadata = preprocess(working_dir)
    for folder_type in ['in', 'ex']:
        flag, common_data = load_common_data(metadata[folder_type]['working_dir'])
        if flag:
            plot_convex = convex(metadata[folder_type]['working_dir'], metadata[folder_type]['lungmask_path'], common_data)
            plot_masking = masking(metadata[folder_type]['working_dir'], common_data)

            masking_list = [(i, mask) for i, mask in enumerate(plot_masking)]
            convex_list = [(i, convex) for i, convex in enumerate(plot_convex)]

            predicted_class_zip = predict(metadata[folder_type]['working_dir'])
            filtered_convex_info, filtered_masking_info = filtering(predicted_class_zip, convex_list, masking_list)
            sorted_combined_images_with_index = convert_image(filtered_convex_info, filtered_masking_info)
            
            is_inside_dict = {}
            y_min_dict = {}
            
            colored_image_list = []

            for index, image in sorted_combined_images_with_index:
                log_progress(index, len(sorted_combined_images_with_index), f"Processing Image({index})...")
                hu_image, _ = process_image(image)
                points = extract_points_in_hu_range(hu_image, lower_bound=300, upper_bound=1900)
                labels = cluster_points(points)

                x_mid, y_min = find_x_mid_and_y_min(hu_image)
                is_inside = check_clusters_intersect_with_rectangle(hu_image, hu_image, x_mid, y_min)
                cluster_labels = cluster_points(points)
                cluster_y_min = find_cluster_y_min(points, cluster_labels)
                is_inside_dict, y_min_dict = store_is_inside_and_y_min(index, is_inside_dict, y_min_dict, is_inside, cluster_y_min)
                
                colored_image = image_process_pipeline(hu_image, points, labels, is_inside, index, y_min_dict)

                if image.shape[-1] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                overlay_image = cv2.addWeighted(image, 0.5, colored_image, 0.5, 0)
                colored_image_list.append((index, overlay_image))

                output_dir = os.path.join(args.output_dir, args.patient_id, folder_type)
                save_overlay_image(overlay_image, index, output_dir)
    result_dir = os.path.join(args.output_dir, args.patient_id)
    log("success", f"Process Finished! Check the result in {result_dir}")

if __name__ == '__main__':
    args = init_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)