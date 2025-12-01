# LOL Dataset Setup Guide

## 📁 디렉토리 구조

프로젝트 루트에 다음과 같은 구조로 LOL 데이터셋을 배치하세요:

```
K_LOLV2T/
├── lol-data/
│   ├── annotations/
│   │   ├── training.json
│   │   ├── validation.json
│   │   ├── testing.json
│   │   └── mart_word2idx.json
│   ├── training/
│   │   ├── XXX_YYY-ZZZ_rgb.pkl
│   │   ├── XXX_YYY-ZZZ_flow.pkl
│   │   └── ...
│   ├── validation/
│   │   ├── XXX_YYY-ZZZ_rgb.pkl
│   │   ├── XXX_YYY-ZZZ_flow.pkl
│   │   └── ...
│   └── testing/
│       ├── XXX_YYY-ZZZ_rgb.pkl
│       ├── XXX_YYY-ZZZ_flow.pkl
│       └── ...
├── config/
│   └── caption/
│       └── paper2020/
│           └── lol_mart.yaml
└── ...
```

## 📥 데이터 다운로드 및 설정

### 1. 데이터 다운로드
팀 공유 드라이브에서 다음 파일들을 다운로드:
- `train.tar.gz` (10.6GB)
- `valid.tar.gz` (1.3GB)
- `test.tar.gz` (3.1GB)
- `annotations.tar.gz`

### 2. 압축 해제
```bash
# 프로젝트 루트로 이동
cd /path/to/K_LOLV2T

# lol-data 디렉토리 생성
mkdir -p lol-data

# 데이터 압축 해제
cd lol-data
tar -xzf train.tar.gz
tar -xzf valid.tar.gz
tar -xzf test.tar.gz
tar -xzf annotations.tar.gz
```

### 3. Vocabulary 파일 확인
`lol-data/annotations/mart_word2idx.json` 파일이 있는지 확인하세요.
- 총 8,169개 단어
- LOL 게임 특화 용어 포함 (baron, dragon, inhibitor, tower 등)

## ⚙️ Config 파일 설정

`config/caption/paper2020/lol_mart.yaml`에서 경로가 올바른지 확인:

```yaml
dataset_train:
    name: "lol"
    # ... 

dataset_val:
    same_as: "dataset_train"
    split: "val"
```

## 🚀 학습 실행

### 로컬 테스트 (CPU)
```bash
python train_caption.py -c config/caption/paper2020/lol_mart.yaml --dataset_max 10
```

### Colab GPU 학습
1. Colab Pro+ 환경에서 SSH 연결
2. 데이터를 Colab 환경에 업로드 또는 Google Drive 마운트
3. 학습 실행:
```bash
python train_caption.py -c config/caption/paper2020/lol_mart.yaml
```

## 📊 데이터 통계

- **Training**: 13,954개 클립
- **Validation**: 1,702개 클립  
- **Testing**: 3,790개 클립
- **총 캡션**: 62,677개

## 🔧 트러블슈팅

### 경로 오류
만약 데이터 경로 오류가 발생하면:
```python
# mart/recursive_caption_dataset.py에서 확인
self.annotations_dir / self.dset_name  # annotations/lol
self.video_feature_dir / self.dset_name  # data/mart_video_feature/lol
```

기본 경로는 프로젝트 루트 기준입니다. 다른 위치에 데이터가 있다면:
```bash
python train_caption.py -c config/caption/paper2020/lol_mart.yaml \
    --annotations_dir /your/path/to/annotations \
    --video_feature_dir /your/path/to/video/features
```

### 메모리 부족
배치 사이즈 조정:
```yaml
train:
    batch_size: 8  # 기본값 16에서 줄이기
```

## 💡 팀원 간 데이터 공유

**주의**: `lol-data/` 폴더는 `.gitignore`에 포함되어 Git에 업로드되지 않습니다.

각 팀원은:
1. 이 문서의 지침에 따라 로컬에 데이터 설정
2. 동일한 디렉토리 구조 유지
3. Config 파일만 Git으로 공유
