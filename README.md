# Rib Muscle Segmentation Module

흉근 세그멘테이션 모듈

-   Dicom 파일로부터 흉근을 추출하고 이를 평가하는 모듈입니다.

## 디렉토리 구조

```
.
├── combine.py: 여러 결과를 결합하는 스크립트
├── data: 데이터 디렉토리
│   ├── input: 환자 DICOM 파일을 저장하는 디렉토리
│   ├── label: 라벨 데이터 저장 디렉토리
│   ├── output: 분석 결과 저장 디렉토리
│   └── working: 중간 작업 데이터를 저장하는 디렉토리
├── dist_run.sh: 모든 환자 데이터를 처리하기 위한 Bash 스크립트
├── error.log: 에러 로그 파일
├── log: 실행 로그를 저장하는 디렉토리
├── model: 모델 관련 파일 디렉토리
│   ├── checkpoint.pth: 학습된 모델 가중치 파일
│   └── model.py: 모델 정의 스크립트
├── modules: 주요 모듈 디렉토리
│   ├── analyzer.py: 분석 관련 기능 모듈
│   ├── classifier.py: 분류 관련 기능 모듈
│   ├── extractor.py: 데이터 추출 관련 기능 모듈
│   ├── postprocess.py: 후처리 관련 기능 모듈
│   ├── preprocess.py: 전처리 관련 기능 모듈
│   └── visualizer.py: 시각화 관련 기능 모듈
├── README.md: 프로젝트 설명 파일
├── requirements.txt: 필요한 라이브러리 목록
├── run.py: 메인 실행 스크립트
└── utils: 유틸리티 모듈 디렉토리
    ├── dicomHandler.py: DICOM 파일 처리 모듈
    ├── directoryHandler.py: 디렉토리 관리 모듈
    ├── imageHandler.py: 이미지 처리 모듈
    ├── logger.py: 로깅 관련 모듈
    └── lungmask.py: 폐 마스크 생성 모듈
```

## 환경 설정(콘다 기반)

1. 가상환경 생성:`conda create -n {환경 이름} python=3.12`

2. 가상환경 활성화: `conda activate {환경 이름}`

3. 라이브러리 설치: `pip install -r requirements.txt`

## 데이터 추가

1. `data/input` 디렉토리에 환자 디렉토리 및 DICOM 파일 추가

```
├── data
│   ├── input
│   │   ├── 10003382
│   │   │   ├── IM-0001-0001.dcm
│   │   │   ├── IM-0001-0002.dcm
│   │   │   ├── IM-0001-0003.dcm
        ⇣   ⇣
```

2. (선택 사항) `data/input` 디렉토리에 환자 디렉토리 및 DICOM 파일 추가

    이 데이터는 추출된 흉근에 대한 성능 지표(IoU)를 계산하기 위해 사용됩니다.

```
│   ├── label
│   │   ├── 10003382
│   │   │   ├── exhalation.seg.nrrd: exhalation 라벨 데이터
│   │   │   └── inhalation.seg.nrrd: inhalation 라벨 데이터
        ⇣   ⇣
```

## 사용법

### 특정 환자에 대해 흉근 추출

```
python run.py --patient_id={환자 번호} --overlap --calculate
```

#### 매개 변수

-   `--patient_id={환자 번호}`: 타겟 환자 번호를 설정합니다.
-   `--overlap`: 추출된 흉근 이미지를 원본 이미지에 오버랩하여 저장할지 여부를 설정합니다.(기본값: False)
-   `--calculate`: 추출된 흉근 이미지와 라벨 이미지의 IoU 값 계산 여부를 설정합니다.(기본값: False)

#### 흉근 추출만 진행

`python run.py --patient_id={환자 번호}`

-   추출된 흉근 이미지 저장 경로: `data/working/{환자 번호}/visualize`
-   오버랩된 흉근 이미지 저장 경로: `data/output/{환자 번호}/overlap`
-   슬라이드별 흉근 픽셀수 저장 경로: `data/output/{환자 번호}/count_in.xlsx`, `data/output/{환자 번호}/count_ex.xlsx`

#### 흉근 추출 및 IoU 계산

라벨 데이터가 있는 환자에 대해 IoU 계산이 진행됩니다.

`python run.py --patient_id={환자 번호} --calculate`

-   추출된 흉근 이미지 저장 경로: `data/working/{환자 번호}/visualize`
-   오버랩된 흉근 이미지 저장 경로: `data/output/{환자 번호}/overlap`
-   슬라이드별 흉근 픽셀수 저장 경로: `data/output/{환자 번호}/count_in.xlsx`, `data/output/{환자 번호}/count_ex.xlsx`
-   추출-정답 비교 이미지 저장 경로: `data/output/{환자 번호}/in`, `data/output/{환자 번호}/ex`
-   슬라이드 별 IoU 결과 데이터 저장 경로: `data/output/{환자 번호}/result_in.xlsx`, `data/output/{환자 번호}/result_ex.xlsx`
-   IoU 결과 그래프 저장 경로: `data/output/{환자 번호}/result_graph.png`

### 모든 환자 실행

```
dist_run.ps1 #PowerShell로 실행

or

bash dist_run.sh #Bash Shell로 실행
```

-   실행 로그 저장 경로: `log/{실행 일/시}/{환자 ID}.log`
