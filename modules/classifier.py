import os
import pydicom
import shutil
from utils.dicomHandler import get_dicom_idx
from utils.directoryHandler import create_working_dir_structure
from utils.orderChecker import check_order
from utils.logger import log, log_execution_time, log_summary, console_classify

def copy_to_working_dir(target_dir: str, target_list: list, reversed: bool = False):
    """
    DICOM 파일 리스트를 작업 디렉토리로 복사하는 함수

    Args:
        target_dir (str): 작업 디렉토리 경로
        dicom_files (list): DICOM 파일 리스트
    """
    if reversed:
        target_list = target_list[::-1]
    sorted_dict = {}
    sorted_sl_dict = {}
    count = {}
    for target in target_list:
        filepath, sn, sl, _ = target
        cur_dir = os.path.join(target_dir, str(sn))
        os.makedirs(cur_dir, exist_ok=True)
        if sn not in sorted_dict:
            sorted_dict[sn] = []
            sorted_sl_dict[sn] = []
        if sn not in count:
            count[sn] = 0
        cur_file = os.path.join(cur_dir, f"{count[sn]:03d}.dcm")
        shutil.copy(filepath, cur_file)
        sorted_dict[sn].append(cur_file)
        sorted_sl_dict[sn].append(sl)
        count[sn] += 1
    return sorted_dict, sorted_sl_dict

def matching_keyword(dicom_data: pydicom.dataset.FileDataset):
    """
    DICOM 데이터셋에서 특정 키워드를 찾는 함수

    Args:
        dicom_data (pydicom.dataset.FileDataset): DICOM 데이터셋

    Returns:
    
    """
    if 'SeriesDescription' in dicom_data:
        description = dicom_data.SeriesDescription.lower()
        in_keywords = ['inspiration', 'inhalation', 'insp', 'in', 'ins', 'pre']
        ex_keywords = ['expiration', 'exhalation', 'exp', 'ex', 'exh', 
                        'post', 'out', 'exhale', 'expiratory',
                        'forced exp', 'end exp']
        
        matched_in = [keyword for keyword in in_keywords if keyword in description]
        matched_ex = [keyword for keyword in ex_keywords if keyword in description]

        if matched_in:
            return 'in'
        elif matched_ex:
            return 'ex'
        else:
            return 'in'
    else:
        return 'unknown'
    
def check_axial(dicom_file_1: str, dicom_file_2: str):
    """
    두 DICOM 파일이 Axial 이미지인지 확인하는 함수

    Args:
        dicom_file_1 (str): 첫 번째 DICOM 파일 경로
        dicom_file_2 (str): 두 번째 DICOM 파일 경로

    Returns:
        bool: 두 파일이 Axial 이미지인 경우 True, 그렇지 않으면 False
    """
    ds1 = pydicom.dcmread(dicom_file_1)
    ds2 = pydicom.dcmread(dicom_file_2)
    
    postion_1 = ds1.ImagePositionPatient
    postion_2 = ds2.ImagePositionPatient
    
    gap = [postion_1[0] - postion_2[0],
           postion_1[1] - postion_2[1],
           postion_1[2] - postion_2[2]]
    flag = gap[0] == 0 and gap[1] == 0
    return flag

@log_execution_time
def classify(patient_id: str, input_dir: str, working_dir: str, output_dir: str, console):
    """
    입력 디렉토리의 DICOM 파일들을 분류하는 함수

    Args:
        patient_id (str): 환자 ID
        input_dir (str): 입력 디렉터리 경로
        preprocess_dir (str): 분류된 DICOM 파일들을 저장할 디렉터리 경로

    Returns:
        str: 분류된 DICOM 파일들을 저장한 디렉터리 경로
    """
    classified_list = []
    
    _date_list = []
    _sub_list = []
    _summary_list = []

    cur_patient_dir = os.path.join(input_dir, patient_id)
    date_list = [date for date in os.listdir(cur_patient_dir) if os.path.isdir(os.path.join(cur_patient_dir, date))]
    date_list.sort()
    cur_working_dir = os.path.join(working_dir, patient_id)
    cur_output_dir = os.path.join(output_dir, patient_id)
    os.makedirs(cur_working_dir, exist_ok=True)
    os.makedirs(cur_output_dir, exist_ok=True)

    for date in date_list:
        
        cur_date_dir = os.path.join(cur_patient_dir, date)
        cur_working_date_dir = os.path.join(cur_working_dir, date)
        cur_output_date_dir = os.path.join(cur_output_dir, date)
        os.makedirs(cur_working_date_dir, exist_ok=True)
        os.makedirs(cur_output_date_dir, exist_ok=True)
        
        sub_dir_list = [d for d in os.listdir(cur_date_dir) if os.path.isdir(os.path.join(cur_date_dir, d))]
        
        sub_dir_files = []
        
        for sub_dir in sub_dir_list:
            
            summary = {'in': 0, 'ex': 0, 'skipped': 0, 'error': 0}
            classify_dict = {"series_description": [], "in": [], "ex": []}
            slide_location_dict = {"in": [], "ex": []}
            
            cur_sub_input_dir = os.path.join(cur_date_dir, sub_dir)
            
            files = [os.path.join(cur_sub_input_dir, f) for f in os.listdir(cur_sub_input_dir) if f.endswith('.dcm')]
            
            flag = check_axial(files[0], files[1])
            
            if flag:
                cur_working_sub_dir = os.path.join(cur_working_date_dir, sub_dir)
                cur_output_sub_dir = os.path.join(cur_output_date_dir, sub_dir)
                os.makedirs(cur_working_sub_dir, exist_ok=True)
                os.makedirs(cur_output_sub_dir, exist_ok=True)
                sub_dir_files.extend(files)
            
                size = None
                
                for file in files:
                    if file.endswith('.dcm'):
                        try:
                            dicom_data = pydicom.dcmread(file)
                            sn = str(dicom_data.SeriesNumber)
                            try:
                                sl = dicom_data.SliceLocation
                            except AttributeError:
                                sl = float(dicom_data.ImagePositionPatient[2])
                            # sd = dicom_data.SeriesDescription
                            ps = dicom_data.PixelSpacing
                            st = dicom_data.SliceThickness
                            if size is None:
                                size = (float(ps[0]), float(ps[1]), float(st))
                            cateogry = matching_keyword(dicom_data)
                            if cateogry:
                                if cateogry == 'in':
                                    summary['in'] += 1
                                elif cateogry == 'ex':
                                    summary['ex'] += 1
                                classify_dict[cateogry].append((file, sn, sl, ps))
                            else:
                                summary['skipped'] += 1
                        except:
                            summary['error'] += 1
                            continue
                
                sorted_target_list_in = sorted(classify_dict['in'], key=lambda x: x[2])
                sorted_target_list_ex = sorted(classify_dict['ex'], key=lambda x: x[2])
                order_in = check_order(sorted_target_list_in)
                order_ex = check_order(sorted_target_list_ex)
                sort_flag_in = order_in != 'top'
                sort_flag_ex = order_ex != 'top'
                classify_dict['in'], slide_location_dict['in'] = copy_to_working_dir(cur_working_sub_dir, sorted_target_list_in, sort_flag_in)
                classify_dict['ex'], slide_location_dict['ex'] = copy_to_working_dir(cur_working_sub_dir, sorted_target_list_ex, sort_flag_ex)
                
                _date_list.append(date)
                _sub_list.append(sub_dir)
                _summary_list.append(summary)
                classified_list.append((cur_sub_input_dir, cur_working_sub_dir, cur_output_sub_dir, classify_dict, slide_location_dict, size))
    console_classify(console, _date_list, _sub_list, _summary_list)
    return classified_list