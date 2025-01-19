# Chess Piece Detection using YOLOv5

## 프로젝트 목적
영상에서의 객체 탐지를 공부하기 이전 YOLO의 구조와 알고리즘, 코드 선행을 위한 간단한 수준의 프로젝트 실습.
YOLOv5와 Roboflow의 체스 데이터셋을 이용하여 실습하며, 코드 구조, YOLO 원리, 데이터 수집 및 처리 방법 등을 학습함.
새로운 아이디어를 제시하기보다는 실습에 가까운 프로젝트로 진행.

## 프로젝트 구조
프로젝트 디렉토리의 주요 구성은 아래와 같습니다:

```bash
.
├── models/                # 모델 구성 관련 파일
├── utils/                 # 유틸리티 스크립트
├── data/                  # 데이터셋 구성 파일 (coco.yaml 등)
├── datasets/              # 체스 말 학습 데이터셋 (Roboflow 제공)
├── yolov5_env/            # 환경설정 관련 스크립트
├── detect.py              # 탐지 실행 스크립트
├── train.py               # 학습 실행 스크립트
├── test.py                # 테스트 스크립트
├── tutorial.ipynb         # Jupyter Notebook 실습 파일
└── requirements.txt       # 의존성 패키지 리스트
```

## 환경 설치 및 데이터셋 수집

### 1. YOLOv5, PyTorch 다운로드 및 설치 과정

#### (1) YOLOv5 다운로드
GitHub에서 YOLOv5 저장소를 클론:

```bash
git clone https://github.com/ultralytics/yolov5.git
```
클론한 디렉토리를 작업 디렉토리로 설정:

```bash
cd yolov5
```

#### (2) Python 환경 설정
Python 3.7 이상 필요. Pycharm에서 프로젝트 환경 생성.
`requirements.txt` 파일을 통해 필수 패키지 설치:

```bash
pip install -U -r requirements.txt
```

#### (3) PyTorch 설치
PyTorch 설치 명령:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

#### (4) YOLOv5 테스트 실행
YOLOv5가 제대로 설치되었는지 확인:

```bash
python detect.py --source data/images/zidane.jpg --weights yolov5s.pt --img 640
```

### 2. Roboflow에서 데이터 가져오기 및 적용

#### (1) 데이터셋 다운로드
Roboflow Chess Piece Dataset에서 데이터셋 다운로드.
- 다운로드 시 YOLOv5 형식으로 데이터셋을 설정.
- 데이터셋에 Augmentation 옵션 추가: Rotation, Blur, Noise 등.

#### (2) 데이터셋 디렉토리 구성
다운로드 받은 데이터셋을 `datasets/Chess_Pieces` 디렉토리에 추가.

```plaintext
yolov5/
├── datasets/
│   ├── Chess_Pieces/
│   │   ├── train/  # 학습 데이터
│   │   ├── valid/  # 검증 데이터
│   │   └── test/   # 테스트 데이터
```

#### (3) `data/coco.yaml` 파일 수정
`coco.yaml` 파일을 열어 데이터셋 경로를 Roboflow 데이터셋으로 변경:

```yaml
train: datasets/Chess_Pieces/train
val: datasets/Chess_Pieces/valid
nc: 12
names: ['white-king', 'white-queen', 'white-bishop', 'white-knight', 'white-rook', 'white-pawn',
        'black-king', 'black-queen', 'black-bishop', 'black-knight', 'black-rook', 'black-pawn']
```

## 모델 학습 및 테스트

### 1. 모델 학습
YOLOv5 모델 학습을 위해 아래 명령어를 실행합니다:

```bash
python train.py --img 640 --batch 16 --epochs 50 --data data/coco.yaml --weights yolov5s.pt
```
- `--img`: 입력 이미지 크기 (640x640).
- `--batch`: 배치 크기 (16).
- `--epochs`: 학습 에폭 수 (50).
- `--data`: 학습 데이터 및 클래스 정보 파일 경로.
- `--weights`: 사전 학습된 가중치 파일 (`yolov5s.pt`).

학습 결과:
- 학습 로그와 모델 가중치는 `runs/train/exp/weights/` 디렉토리에 저장.
- 주요 파일:
  - `best.pt`: 학습된 최종 모델.
  - `last.pt`: 마지막 에폭 모델.

### 2. 모델 테스트
학습된 모델로 테스트:

```bash
python test.py --data data/coco.yaml --weights runs/train/exp/weights/best.pt --img 640
```
- `--weights`: 학습된 모델 가중치 (`best.pt`).
- `--img`: 입력 이미지 크기.

테스트 결과:
- 테스트 결과는 `runs/test/` 디렉토리에 저장.
- 결과 파일: `results.txt`.

### 3. Inference (탐지 실행)
테스트 이미지에서 탐지 실행:

```bash
python detect.py --source datasets/Chess_Pieces/test --weights runs/train/exp/weights/best.pt --img 640
```
- `--source`: 테스트할 이미지 경로.
- `--weights`: 학습된 모델 가중치.

탐지 결과:
- 탐지 결과 이미지는 `runs/detect/` 디렉토리에 저장.

### 정리된 디렉토리 구조

```plaintext
yolov5/
├── datasets/
│   ├── Chess_Pieces/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
├── runs/
│   ├── train/
│   │   └── exp/
│   │       ├── weights/
│   │       │   ├── best.pt
│   │       │   └── last.pt
│   └── detect/
│       ├── image1.jpg
│       └── image2.jpg
```

## 학습 결과

### 모델 성능
- `mAP_0.5`: 98.4%
- `mAP_0.5:0.95`: 78.0%

### 학습 및 테스트 결과
- 학습 중 손실 그래프
- 테스트 이미지에서 탐지된 체스 말 (샘플 결과 이미지 첨부 예정)

## 프로젝트 중 배운 점과 도전 과제

### 배운 점
- YOLOv5의 구조 및 객체 탐지 모델의 학습 과정 이해.
- 데이터셋 준비와 Augmentation의 중요성 학습.
- Git과 Roboflow 활용법 익히기.

### 도전 과제
- Git 사용 및 불필요한 파일 관리 (.gitignore 등).
- 객체 탐지의 세부적인 코드 이해.
- 다음 프로젝트(예: 축구 경기 객체 탐지)에서 더 많은 실험 및 아이디어 적용.

## 참고 자료
- [YOLOv5 공식 문서](https://github.com/ultralytics/yolov5)
- [Roboflow Chess Piece Dataset](https://roboflow.com/)
