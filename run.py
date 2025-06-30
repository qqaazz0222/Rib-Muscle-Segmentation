import os
import json
import torch
import pickle
import multiprocessing
from argparse import ArgumentParser
from rich.console import Console
from model.model import init_model, predict
from modules.classifier import classify
from modules.postprocess import post_processing
from modules.extractor_optimized import extract, calc_boundary
from modules.analyzer import load_label, calculate, overlap, counting
from modules.visualizer import visualize
from utils.directoryHandler import check_checkpoint
from utils.dicomHandler import *
from utils.imageHandler import *
from utils.logger import log, log_execution_time, log_execution_time_with_dist, log_progress, console_banner, console_args

@log_execution_time
def init_args(console):
    """
    인자 초기화 함수

    Returns:
        args (ArgumentParser): 인자
    """
    parser = ArgumentParser()
    parser.add_argument('-P', '--patient_id', type=str, default="10003382", help='환자 아이디(단일 환자만 분석)')
    parser.add_argument('-O', '--overlap', action='store_true', help='근육 오버랩 여부')
    parser.add_argument('-C', '--calculate', action='store_true', help='IoU 계산 여부')
    parser.add_argument('-D', '--data_dir', type=str, default='data', help='데이터 디렉토리')
    parser.add_argument('-V', '--visualize', type=bool, default=False, help='시각화 여부')
    parser.add_argument('--batch_size', type=int, default=1, help='한번에 분석할 슬라이드 개수')
    parser.add_argument('--input_dir', type=str, default='data/input', help='입력(원본데이터) 디렉토리')
    parser.add_argument('--output_dir', type=str, default='data/output', help='출력 디렉토리')
    parser.add_argument('--working_dir', type=str, default='data/working', help='작업 디렉토리')
    parser.add_argument('--label_dir', type=str, default='data/label', help='라벨(정답데이터) 디렉토리')
    parser.add_argument('--weight', type=str, default='model/checkpoint.pth', help='모델 가중치 파일 경로')
    parser.add_argument('--threshold', type=float, default=0.9, help='CNN 모델의 확률 임계값')
    args = parser.parse_args()
    dict_args = vars(args)
    console_args(console, dict_args)
    return parser.parse_args()

def extract_helper(params):
    idx, working_file, method, working_file_num, category, sn, boundary, visualize_dir = params
    norm_idx = idx / working_file_num
    original_image, extracted_muscle_image, metadata = extract(idx, norm_idx, category, sn, method, boundary, working_file, visualize_dir)
    return working_file, original_image, extracted_muscle_image, metadata

def extract_muscle(working_list: list, pred_list: list, category: str, sn: str, checkpoint_dir: str, visualize_dir: str, batch_size: int = 1):
    """
    근육 추출 함수

    Args:
        working_list (list): 작업 리스트
        pred_list (list): 예측된 클래스 리스트
        category (str): 분류 카테고리
        sn (str): 시리즈 번호
        checkpoint_dir (str): 체크포인트 디렉토리
        visualize_dir (str): 시각화 디렉토리

    Returns:
        list: 원본 이미지 리스트
        list: 추출된 근육 이미지 리스트
    """
    st = log_execution_time_with_dist("start", "extract_muscle")
    checkpoint = f"checkpoint_extract_{category}_{sn}.pkl"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    
    if check_checkpoint(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)

    original_image_list = []
    extracted_muscle_image_list = []
    metadata_list = []
    working_file_num = len(working_list) - 1
        
    boundary = calc_boundary(working_list)
    
    param_list = []
    for idx, working_file in enumerate(working_list):
        method = pred_list[idx]
        param_list.append((idx, working_file, method, working_file_num, category, sn, boundary, visualize_dir))

    with multiprocessing.Pool(processes=batch_size) as pool:
        for i, (working_file, original_image, extracted_muscle_image, metadata) in enumerate(pool.imap(extract_helper, param_list)):
            log_progress(i + 1, len(working_list), f"Extracting Muscle ({working_file})")
            original_image_list.append(original_image)
            extracted_muscle_image_list.append(extracted_muscle_image)
            metadata_list.append(metadata)

    with open(checkpoint_path, "wb") as f:
        pickle.dump([original_image_list, extracted_muscle_image_list, metadata_list], f)
        
    log_execution_time_with_dist("end", "extract_muscle", st)
    return original_image_list, extracted_muscle_image_list, metadata_list

def main():
    """
    메인 함수

    Args:
        args (ArgumentParser): 인자

    Returns:
        None
    """
    console = Console()
    console = console.__class__(log_time=False)
    console_banner(console)
    
    args = init_args(console)
    if args.data_dir != 'data':
        args.input_dir = os.path.join(args.data_dir, 'input')
        args.output_dir = os.path.join(args.data_dir, 'output')
        args.working_dir = os.path.join(args.data_dir, 'working')
        args.label_dir = os.path.join(args.data_dir, 'label')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = init_model(args.weight, device)

    classified_list = classify(args.patient_id, args.input_dir, args.working_dir, args.output_dir, console)
    for cur_sub_input_dir, working_dir, output_dir, classify_dict, slide_location_dict, size in classified_list:
        date = working_dir.split('/')[-2]
        log("info",  f"Processing Patient ID: {args.patient_id}, Date: {date}", upper_div=True, space=True)
        label_dir = os.path.join(args.label_dir, args.patient_id)
        # output_dir = os.path.join(args.output_dir, args.patient_id)
        checkpoint_dir = os.path.join(working_dir, 'checkpoint')
        visualize_dir = os.path.join(working_dir, 'visualize')
        muscle_dir = os.path.join(output_dir, 'muscle')
        abs_dir = os.path.join(output_dir, 'abs')
        overlap_dir = os.path.join(output_dir, 'overlap')
        
        original_image_dict = {"in": {}, "ex": {}}
        extracted_muscle_image_dict = {"in": {}, "ex": {}}
        metadata_dict = {"in": {}, "ex": {}}
        
        for category in ["in", "ex"]:
            sn_list = classify_dict[category].keys()
            for sn in sn_list:
                working_list = classify_dict[category][sn]
                pred_list = predict(model, transform, device, checkpoint_dir, working_list, category, sn)
                if pred_list is not None and pred_list[0] == 'lower' and pred_list[-1] == 'upper':
                    working_list.sort(reverse=True)
                    pred_list.sort(reverse=True)
                original_image_list, extracted_muscle_image_list, metadata_list = extract_muscle(working_list, pred_list, category, sn, checkpoint_dir, visualize_dir, batch_size = args.batch_size)
                original_image_dict[category][sn] = original_image_list
                extracted_muscle_image_dict[category][sn] = extracted_muscle_image_list
                metadata_dict[category][sn] = metadata_list
            
        metadata_path = os.path.join(working_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=4)
            
        post_processed_muscle_image_dict = post_processing(extracted_muscle_image_dict, metadata_dict)
        
        counting(post_processed_muscle_image_dict, slide_location_dict, cur_sub_input_dir, output_dir, args.output_dir, size, args.visualize)
        if args.overlap:
            overlap(original_image_dict, post_processed_muscle_image_dict, muscle_dir, abs_dir, overlap_dir)
        if args.calculate:
            label_muscle_image_dict, label_flag = load_label(checkpoint_dir, label_dir)
            if label_flag:
                calculate(post_processed_muscle_image_dict, label_muscle_image_dict, output_dir)
                visualize(output_dir, (0, 1))
        else:
            log("dimmed", "Skip Analysis")
    log("success",  f"Processing Finished")

if __name__ == '__main__':
    main()