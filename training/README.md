# GPU 추가 학습: 강의실 코드 인식

이번 실험은 과목명/시간표 전체 모델을 새로 학습한 것이 아니라, 가장 약했던 **강의실 코드용 시각 OCR**을 추가 학습한 결과입니다. 기존 Tesseract 과목명·격자·시간 처리는 동결했습니다. 기존 브라우저 UI와 원래 사이트에 이 모델을 연결하는 작업은 아직 하지 않았습니다.

## 비교 결과

최종 평가 사진 2장, 코드가 보이는 수업 24개에서 다음 결과를 얻었습니다.

| 항목 | 기존 Tesseract 프로그램 | 새 모델 학습 전 | 같은 모델 학습 후 |
|---|---:|---:|---:|
| 강의실 코드 전체 일치 | 8/24 (33.3%) | 17/24 (70.8%) | **23/24 (95.8%)** |
| 문자 오류율 CER | 18.18% | 11.36% | **1.14%** |
| 과목명·강의실·요일·시간 모두 일치 | 7/24 (29.2%) | 13/24 (54.2%) | **18/24 (75.0%)** |

학습 전후 비교는 **동일한 입력 자르기·동일한 구조·동일한 디코더**를 사용하고 가중치만 바꿨습니다. 따라서 모델 교체 효과와 추가 학습 효과를 구분합니다. 기존 전체 8장 평가의 50.6%와 이번 2장 평가를 직접 비교하면 안 됩니다.

학습 데이터에 같은 코드 문자열이 없었던 항목 18개만 보면 **16/18 (88.9%) → 17/18 (94.4%)**입니다. 24개 전체보다 개선 폭이 작습니다. 평가에 나타나는 코드 문자열은 11종이고 수업 간 중복이 있으므로 독립된 24개 새 환경이라고 볼 수 없습니다.

남은 오류는 `P305 → P30E`입니다. 평가 후 이 오류에 맞춘 치환 규칙이나 재학습은 추가하지 않았습니다. 과목명 정확도는 이번 변경 범위 밖이므로 그대로 18/24입니다.

## 과적합·데이터 누출 방지

- 이미지 단위 분할: 학습 1·2·4·5·6, 검증 3, 최종 평가 7·8.
- 관련 시간표 1/2, 4/5, 7/8은 각각 같은 분할에 둡니다. 첨부 중 원본 중복 한 장은 이미 제거했습니다.
- 같은 픽셀의 학습 crop 중복을 제거하고 학습/검증·평가 간 같은 crop이 없도록 SHA-256으로 검사했습니다.
- 학습 crop 2개가 잘못 잘린 것을 시각 점검으로 찾아 제외했습니다. 최종 학습 입력은 **32개**, 검증 9개입니다.
- 외부 이미지를 새 학습 데이터로 추가하지 않았습니다. 회전·흐림·굵기·명암·크기·잡음 증강은 학습 이미지에만 적용합니다.
- 공개 사전학습 모델 EasyOCR `english_g2`를 사용하고 CNN 및 첫 LSTM을 고정했습니다. 새로 학습한 데이터만 제공 사진으로 제한한 것이며, 사전학습 모델까지 이 사진만으로 만들어진 것은 아닙니다.
- AdamW, 낮은 학습률 3e-5, weight decay, gradient clipping, 최대 25 epoch, 검증 기반 patience 5를 적용했습니다.
- 검증 정확도 → CER → CTC loss 순으로 checkpoint를 선택합니다. 15 epoch를 선택했고 20 epoch에서 중단했습니다. 최종 평가로 checkpoint를 고르지 않았습니다.
- 첫 실험의 잘못된 학습 crop을 발견해 해당 run을 폐기했습니다. 검증 점수가 처음부터 9/9라 CTC loss를 tie-breaker로 추가한 프로토콜 v2를 최종 평가 전에 고정했습니다. 최초 run의 근거는 로컬 `training/private/initial-data-audit-run/`에 보관했습니다.

**과적합이 없다고 증명한 것은 아닙니다.** 원본 8장은 이전 파서 개발에서 이미 살펴봤고, 이번 분리는 추가 학습 과정에 대해서만 적용됩니다. 검증 1장·최종 평가 2장뿐이므로 새로운 학교, 글꼴, 테마, 앱 화면에서 같은 성능을 보장할 수 없습니다. 다음 단계에는 이번에 전혀 사용하지 않은 실제 사진을 모아 평가해야 합니다.

## 모델과 실행

- 모델: VGG + BiLSTM + CTC, 3,781,345 parameters.
- 학습 GPU: RTX 2070, PyTorch 2.6.0+cu124, CUDA 12.4.
- 실제 최종 학습 루프: 약 24.8초. 환경 설치·사진 처리 시간 제외.
- 내보낸 모델: [`../trained/room-code-v2.onnx`](../trained/room-code-v2.onnx), 약 15.1 MB.
- ONNX Runtime CPU에서도 평가 24개의 예측이 PyTorch와 모두 같음을 확인했습니다.
- **브라우저·모바일 실행은 아직 연결/측정하지 않았습니다.** 이 결과를 웹 실측 결과로 해석하면 안 됩니다.

## 재현

프로젝트 루트에서 실행합니다. 제공 사진과 정답은 공개 저장소에 포함하지 않았으므로 `samples/private/`의 원본·truth·기존 baseline 결과가 필요합니다.

```powershell
python -m venv .venv-train
./.venv-train/Scripts/python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
./.venv-train/Scripts/python.exe -m pip install -r training/requirements.txt
node training/prepare.mjs
./.venv-train/Scripts/python.exe training/setup.py
./.venv-train/Scripts/python.exe training/train.py
./.venv-train/Scripts/python.exe training/evaluate.py
./.venv-train/Scripts/python.exe training/export.py
./.venv-train/Scripts/python.exe training/test_training.py
```

`train.py`는 최종 평가 파일을 열지 않습니다. `evaluate.py`는 프로토콜과 선택된 checkpoint의 SHA-256을 확인한 뒤 최종 평가를 수행합니다. `protocol.json`, `evaluation/finetune-history.json`, `evaluation/finetune-comparison.json`에 설정·곡선·집계를 보관했습니다. 세부 원문과 사용자 사진은 `training/private/`에만 있습니다.

## 출처 및 라이선스

모델 구조는 [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)의 `easyocr/model/vgg_model.py`와 `modules.py`를 바탕으로 필요한 부분을 구현했습니다. 사전학습 가중치의 공식 URL과 checksum은 `setup.py` 및 로컬 `pretrained-source.json`에 기록했습니다. 원본은 Apache-2.0이며 해당 라이선스는 `../public/licenses/EasyOCR-APACHE-2.0.txt`에 포함합니다.
