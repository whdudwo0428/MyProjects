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

### 1. 환경 설치
Python 3.7 이상 필요.
아래 명령어로 필수 패키지를 설치합니다:

```bash
pip install -U -r requirements.txt
```

### 2. 데이터셋 수집
데이터셋은 Roboflow Chess Piece Dataset에서 제공됩니다.

- 데이터셋 Augmentation: Rotation, Blur, Noise 등의 저수준 Augmentation을 추가로 적용하여 학습 성능을 개선했습니다.

## 모델 학습 및 테스트

### 1. 모델 학습
YOLOv5 모델 학습을 위해 아래 명령어를 실행합니다:

```bash
python train.py --img 640 --batch 16 --epochs 50 --data data/coco.yaml --weights yolov5s.pt
```
- `--img`: 입력 이미지 크기 (픽셀 단위)
- `--batch`: 배치 크기
- `--epochs`: 학습 에폭 수
- `--data`: 데이터셋 구성 파일
- `--weights`: 사전 학습된 모델 가중치 파일

### 2. 모델 테스트
학습된 모델의 성능을 평가하려면 아래 명령어를 실행합니다:

```bash
python test.py --data data/coco.yaml --weights runs/train/exp/weights/best.pt --img 640
```

### 3. 모델 탐지
임의의 이미지에서 체스 말을 탐지하려면:

```bash
python detect.py --source datasets/Chess_Pieces/test --weights runs/train/exp/weights/best.pt --img 640
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
