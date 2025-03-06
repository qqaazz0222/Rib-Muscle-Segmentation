# 현재 시간 저장
curTime=$(date +"%Y%m%d_%H%M%S")
# log 디렉토리 생성
mkdir -p log
# log 디렉토리에 현재 시간을 이름으로하는 디렉토리 생성
mkdir -p log/$curTime
# 해당 싱행에 대해 로그를 저장할 디렉토리 경로를 저장
log_dir=log/$curTime
# data/input 디렉토리에서 환자 목록 불러오기 (디렉토리만 포함)
patients=$(find data/input -maxdepth 1 -type d -exec basename {} \; | tail -n +2)
# 환자 목록을 하나씩 출력하며 해당 환자를 분석
for patient in $patients; 
do 
    echo "Current Patient ID: $patient"; 
    echo "Starting analysis for patient $patient"
    # 환자id를 인자로 하는 스크립트 실행
    nohup python run.py --patient_id=$patient > $log_dir/$patient.log 2>&1 &
done