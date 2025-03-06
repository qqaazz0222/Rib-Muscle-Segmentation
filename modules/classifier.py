import os
import pydicom
import shutil
from utils.directoryHandler import create_working_dir_structure
from utils.logger import log_execution_time, log_summary

@log_execution_time
def classify(patient_id: str, input_dir: str, preprocess_dir: str):
    """
    입력 디렉토리의 DICOM 파일들을 분류하는 함수

    Args:
        patient_id (str): 환자 ID
        input_dir (str): 입력 디렉터리 경로
        preprocess_dir (str): 분류된 DICOM 파일들을 저장할 디렉터리 경로

    Returns:
        str: 분류된 DICOM 파일들을 저장한 디렉터리 경로
    """
    summary = {'in': 0, 'ex': 0, 'skipped': 0, 'error': 0}

    cur_patient_dir = os.path.join(input_dir, patient_id)
    cur_preprocess_dir, cur_in_dir, cur_ex_dir = create_working_dir_structure(patient_id, preprocess_dir)

    for root, _, files in os.walk(cur_patient_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.dcm'):
                if file == 'output_HU_in.dcm' or file == 'output_HU_ex.dcm':
                    target_path = os.path.join(cur_preprocess_dir, file)
                    shutil.copy(file_path, target_path)
                else:
                    try:
                        dicom_data = pydicom.dcmread(file_path)
                        cateogry = matching_keyword(dicom_data)
                        if cateogry:
                            if cateogry == 'in':
                                summary['in'] += 1
                                target_dir = cur_in_dir
                            else:
                                summary['ex'] += 1
                                target_dir = cur_ex_dir
                            target_path = os.path.join(target_dir, file)
                            shutil.copy(file_path, target_path)
                        else:
                            summary['skipped'] += 1
                    except:
                        summary['error'] += 1
                        continue
                    
    log_summary(summary)
    return cur_preprocess_dir                 
    
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
            return 'unknown'
    else:
        return 'unknown'