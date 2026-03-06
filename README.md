# GeoDANO

지오메트리 도메인 적응 및 수학적 시각 이해를 위한 GeoDANO 레포입니다. 아래 가이드는 체크포인트/설치/추론/학습 절차를 한 곳에서 정리합니다.

## 1. Checkpoint 다운로드

<!-- 작성 예정 -->

## 2. Installation

사전 준비: Python 3.10+, CUDA 12.1 환경을 권장합니다.

1) PyTorch 설치

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

2) GeoCLIP 설치 (GeoCLIP/README.md 참고)

로컬 소스에서 editable 모드로 설치합니다. 훈련 옵션 포함 설치를 권장합니다.

```bash
# 레포 루트에서 실행
pip install -e "GeoCLIP[training]"
```

3) GeoDANO(LLaVA 기반) 설치 (GeoDANO/README.md 참고)

로컬 소스에서 editable 모드로 설치합니다.

```bash
# 레포 루트에서 실행
# 추론만 필요한 경우: standalone, 학습까지 포함: train
pip install -e "GeoDANO[train]"
# 또는 (추론 전용)
# pip install -e "GeoDANO[standalone]"
```

설치가 완료되면 `llava` 및 `open_clip_torch` 모듈이 동일한 가상환경에서 import 가능해야 합니다.

## 3. Inference

1) MathVerse 다운로드

<!-- 작성 예정 -->

2) `inference.py` 실행

아래 예시는 기본 LLaMA-3 8B 기반 체크포인트를 사용해 MathVerse 형식의 테스트 JSON에 대해 응답을 생성합니다. 경로는 환경에 맞게 변경하세요.

```bash
python inference.py \
  --pretrained /path/to/ckpt_dir \
  --model-base meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset-json /path/to/MathVerse/test.json \
  --images-dir /path/to/MathVerse/images \
  --output results.jsonl \
  --device cuda \
  --device-map auto \
  --conv-template llava_llama_3
```

- `--pretrained`: 학습 완료(또는 배포)된 LLaVA 가중치 디렉토리/허브 ID
- `--dataset-json`: MathVerse 테스트 JSON 경로
- `--images-dir`: JSON에 명시된 이미지 루트 디렉토리
- 결과는 `--output` 경로에 JSON 리스트 형태로 저장됩니다.

3) `evaluation.py` 실행

<!-- 작성 예정 -->

## 4. Training

1) Training data 다운로드

<!-- 작성 예정 -->

2) 학습 실행 (GeoDANO/scripts/train/direct_finetune_gps_local.sh 참고)

로컬 환경에서 디스트리뷰티드/DeepSpeed 설정 없이 바로 동작하도록 구성된 스크립트입니다. 필수 인자는 아래와 같습니다.

- `--dataset-json`: 학습용 JSON (LLaVA 포맷)
- `--images-dir`: 이미지 루트 디렉토리
- `--vision-encoder-path`: 사전학습 비전 인코더 가중치(예: GeoCLIP_DAv2.pt)

사용 예시:

```bash
bash GeoDANO/scripts/train/direct_finetune_gps_local.sh \
  --dataset-json /data/gps_program_inst_train_rich.json \
  --images-dir /data/GPS_Program \
  --vision-encoder-path /checkpoints/GeoCLIP_DAv2.pt \
  --num-gpus 2 \
  --report-to none
```

주요 선택 인자:
- `--llm`: 기본 LLM ID (기본값: `meta-llama/Meta-Llama-3-8B-Instruct`)
- `--vision-model`: 비전 타워 ID (기본값: `open_clip_hub:ViT-L-14-336`)
- `--prompt-template`: 대화 템플릿 키 (기본값: `llava_llama_3`)
- `--no-deepspeed`: 지정 시 DeepSpeed 비활성화 (기본은 사용 가능 시 활성)

출력: 체크포인트 및 로그는 기본적으로 `./checkpoints/<자동 생성된 러닝 이름>/`에 저장됩니다. DeepSpeed 설정은 `GeoDANO/scripts/zero2.json` 등에서 조정할 수 있습니다.

---

이 README는 GeoCLIP/GeoDANO 하위 README의 세부 내용을 요약한 상위 가이드입니다. 환경/데이터/체크포인트 경로는 사용 환경에 맞게 수정해 주세요.

