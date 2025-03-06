import os
import cv2
import nrrd
import numpy
import pickle
import slicerio
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from openpyxl import Workbook
from utils.directoryHandler import check_processed_data
from utils.logger import *

@log_execution_time
def init_args():
    """
    인자 초기화 함수

    Returns:
        args (ArgumentParser): 인자
    """
    parser = ArgumentParser()
    parser.add_argument('--patient_id', type=str, default="10003382", help='환자 아이디(단일 환자만 분석)')
    parser.add_argument('--output_dir', type=str, default='data/output', help='출력 디렉토리')
    parser.add_argument('--working_dir', type=str, default='data/working', help='작업 디렉토리')
    parser.add_argument('--label_dir', type=str, default='data/label', help='라벨(정답데이터) 디렉토리')
    return parser.parse_args()

@ log_execution_time
def load_predict_result(output_dir: str, patient_id: str):
    """
    예측 결과를 로드하는 함수

    Args:
        output_dir (str): 출력 디렉터리 경로
        patient_id (str): 환자 ID

    Returns:
        dict: 타입별 이미지 리스트
    """
    predict_data = {
        "in": [],
        "ex": []
    }
    for folder_type in ['in', 'ex']:
        path = os.path.join(output_dir, patient_id, folder_type, "colored_image_list.pkl")
        if os.path.exists(path):
            raw_data = pickle.load(open(path, 'rb'))
            for data in raw_data:
                idx, image = data
                predict_data[folder_type].append(image)
    return predict_data

@ log_execution_time
def load_nrrd_muscle_only(label_dir: str, patient_id: str):
    """
    NRRD 파일에서 Muscle 영역만 추출하는 함수

    Args:
        path (str): NRRD 파일 경로

    Returns:
        list: Muscle 영역 이미지 데이터
    """
    def check_rotate(ratio, image):
        """
        이미지 회전 여부를 체크하는 함수

        Args:
            ratio (float): 비율
            image (numpy.ndarray): 이미지

        Returns:
            numpy.ndarray: 회전된 이미지
        """
        if ratio < 0.4 or ratio > 0.6:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image

    target_dir = os.path.join(label_dir, patient_id)
    save_file = "label_image_muscle_only.pkl"
    save_path = os.path.join(target_dir, save_file)

    if check_processed_data(save_path):
        log("info", f"처리된 데이터가 존재합니다. 해당 데이터를 불러옵니다. ({save_path})")
        return pickle.load(open(save_path, 'rb'))

    nrrd_file = None
    for file_name in os.listdir(target_dir):
        if file_name.endswith('.nrrd'):
            nrrd_file = file_name
            break
    if nrrd_file is None:
        raise FileNotFoundError("No .nrrd file found in the target directory")
    path = os.path.join(target_dir, nrrd_file)
    target = []
    segmentation = slicerio.read_segmentation(path)
    for segment in segmentation['segments']:
        if segment['name'].lower() in ["muscle", "artery"]:
            target.append((segment['name'], segment['labelValue']))
            break
    if len(target) > 0:
        extracted_segmentation = slicerio.extract_segments(segmentation, target)
        voxels = extracted_segmentation['voxels']
        height, width, num_slide = voxels.shape
        label_image_data = []
        for idx in range(num_slide):
            log_progress(idx + 1, num_slide, "Muscle 영역 추출 중...")
            image = numpy.array([[[0,0,0] for _ in range(width)] for _ in range(height)])
            count = {"left": 0, "right": 0}
            for y in range(height):
                for x in range(width):
                    flag = voxels[x, y, idx] == 1
                    if flag:
                        if x < width // 2:
                            count["left"] += 1
                        else:
                            count["right"] += 1
                        image[y, x] = [255, 0, 0]
                    else:
                        image[y, x] = [0, 0, 0]
            ratio = count["left"] / (count["left"] + count["right"])
            image = check_rotate(ratio, image)
            label_image_data = [image] + label_image_data
        pickle.dump(label_image_data, open(os.path.join(target_dir, "label_image_muscle_only.pkl"), 'wb'))
        return label_image_data
    else:
        raise ValueError("No muscle segmentation found")
    
def extract_mucle_area(image: numpy.ndarray):
    """
    이미지에서 Muscle 영역만 추출하는 함수

    Args:
        image (numpy.ndarray): 이미지

    Returns:
        numpy.ndarray: Muscle 영역 이미지 데이터
    """
    height, width, _ = image.shape
    muscle_image = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    muscle_point_min = 100 #근육 영역 g값의 최소값
    muscle_point_max = 255 #근육 영역 g값의 최대값
    for y in range(height):
        for x in range(width):
            _, g, _ = image[y, x, :]
            if g > muscle_point_min and g < muscle_point_max:
                muscle_image[y, x] = [255, 0, 0]
            else:
                muscle_image[y, x] = [0, 0, 0]            
    return muscle_image

def calculate_iou(predict_image: numpy.ndarray, label_image: numpy.ndarray):
    """
    IoU 계산 함수

    Args:
        predict_image (numpy.ndarray): 예측 이미지
        label_image (numpy.ndarray): 라벨 이미지

    Returns:
        float: IoU 값
    """
    intersection = numpy.logical_and(predict_image, label_image)
    union = numpy.logical_or(predict_image, label_image)
    iou_score = numpy.sum(intersection) / numpy.sum(union)
    iou_score = round(iou_score, 2)
    return iou_score


def main(args):
    predict_data = load_predict_result(args.output_dir, args.patient_id)
    label_data = load_nrrd_muscle_only(args.label_dir, args.patient_id)

    wb = Workbook()
    sheet = wb.active
    sheet.append(["Slice", "IoU (In)", "IoU (Ex)", "IoU (Total)"])
    num_slice = min(len(predict_data['in']), len(label_data))
    summary = {
        "Num Slice": num_slice,
        "Num Inference(in)": len(predict_data['in']),
        "Num Inference(ex)": len(predict_data['ex']),
        "Num Label": len(label_data),
        "Avg IoU (In)": 0,
        "Avg IoU (Ex)": 0,
        "Avg IoU (Total)": 0
    }
    iou_in_list = []
    iou_ex_list = []
    iou_total_list = []
    
    for idx in range(num_slice):
        log_progress(idx + 1, num_slice, "Slice 별 IoU 계산 중...")
        label_image = label_data[idx]
        flag = [False, False]
        if len(predict_data['in']) > idx:
            in_image = predict_data['in'][idx]
            in_muscle_image = extract_mucle_area(in_image)
            iou_in = calculate_iou(in_muscle_image, label_image)
            iou_in_list.append(iou_in)
            flag[0] = True
        else:
            iou_in = "N/A"
        if len(predict_data['ex']) > idx:
            ex_image = predict_data['ex'][idx]
            ex_muscle_image = extract_mucle_area(ex_image)
            iou_ex = calculate_iou(ex_muscle_image, label_image)
            iou_ex_list.append(iou_ex)
            flag[1] = True
        else:
            iou_ex = "N/A"
            
        if flag[0] and flag[1]:
            iou_total = (iou_in + iou_ex) / 2
        elif flag[0]:
            iou_total = iou_in
        elif flag[1]:
            iou_total = iou_ex
        
        if flag[0] or flag[1]:
            iou_total_list.append(iou_total)

        sheet.append([idx, iou_in, iou_ex, iou_total])

    summary["Avg IoU (In)"] = numpy.mean(iou_in_list)
    summary["Avg IoU (Ex)"] = numpy.mean(iou_ex_list)
    summary["Avg IoU (Total)"] = numpy.mean(iou_total_list)

    sheet.append(["", summary["Avg IoU (In)"], summary["Avg IoU (Ex)"], summary["Avg IoU (Total)"]])
    wb.save(os.path.join(args.output_dir, args.patient_id, "iou_result.xlsx"))
    log_summary(summary)

if __name__ == '__main__':
    args = init_args()
    main(args)