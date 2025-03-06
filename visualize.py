import os
import cv2
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
    parser.add_argument('--visualized_dir', type=str, default='data/visualiszed', help='시각화 디렉토리')
    return parser.parse_args()

@log_execution_time
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

def main(args: ArgumentParser):


if __name__ == '__main__':
    args = init_args()
    main(args)