import os
from utils.directoryHandler import check_checkpoint
from utils.dicomHandler import get_dicom_files
from utils.lungmask import generate_lungmask
from utils.logger import log_execution_time

@log_execution_time
def create_lung_mask(working_dir: str):
    """
    전처리를 수행하는 함수

    Args:(
        working_dir (str): 작업 디렉터리 경로(./data/working/{환자 ID})

    Returns:
        dict: 전처리 메타데이터
    """
    metadata = {
        "in": {
            "working_dir": os.path.join(working_dir, "in"),
            "lungmask_path": os.path.join(working_dir, "output_HU_in.dcm")
        },
        "ex": {
            "working_dir": os.path.join(working_dir, "ex"),
            "lungmask_path": os.path.join(working_dir, "output_HU_ex.dcm")
        }
    }

    for folder_type in ['in', 'ex']:
        lungmask_path = os.path.join(working_dir, f"output_HU_{folder_type}.dcm")
        if not check_checkpoint(lungmask_path):
            generate_lungmask(working_dir, lungmask_path)
    return metadata