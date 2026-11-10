# Timetable Recognizer

에브리타임 시간표 스크린샷에서 과목명, 강의실, 요일과 시간을 추출해 webWallpaperTimetable에 연결하기 위한 독립 프로젝트입니다.

현재 상태: 설계 단계. OCR 실행 코드와 학습 모델은 아직 포함하지 않습니다.

## 제안 구조

1. 시간표 영역 및 격자선 탐지: 밝은/어두운 테마, 상단 잘림, 하단 오버레이를 처리합니다.
2. 요일과 시간 눈금 OCR 및 좌표 보정: 화면 상태바 시각은 사용하지 않습니다.
3. 색칠된 강의 블록 분리: 같은 색의 인접 블록은 경계선과 텍스트 위치로 분리합니다.
4. 블록별 한국어/영어 OCR: 과목명 줄바꿈 복원, 교수명과 강의실 분리, 원문 보존.
5. 블록 좌표를 요일 및 시간으로 변환하고 불확실성을 표시합니다.
6. 사용자 검토 후 기존 사이트의 classes 형태로 변환합니다.

제안 출력 필드: name, room, day, startMinutes, endMinutes, bounds, rawText, warnings.
기존 사이트 어댑터: id, name, room, day, startH, startM, endH, endM.
시간은 모델이 생성하는 대신 눈금 및 블록 경계에서 계산합니다. 30분 간격으로 무조건 반올림하지 않습니다.

## 실행 방식

- 1차 비교 후보: 브라우저 Tesseract.js(한국어+영어)와 서버 PaddleOCR 한국어 모델.
- 초기 학습은 하지 않고 사전학습 공개 모델과 직접 작성하는 격자/시간/필드 파서를 조합합니다.
- 브라우저 OCR은 가능하지만 모바일 지연, 최대 메모리, 최초 다운로드 크기를 측정해야 합니다.
- PaddleOCR를 브라우저에서 사용하려면 한국어 모델 지원, ONNX 변환, 전후처리 및 런타임 호환성을 별도로 검증합니다.
- 최종 서버 필요 여부와 모델 선택은 동일한 정답 데이터에 대한 실측 후 결정합니다.

## 예외 및 평가

텍스트 줄바꿈, 교수명 유무, 영문/숫자 강의실, 오전/오후, 30분 경계,
스크롤로 잘린 블록, 하단 가림, 시간 미지정 과목, 다양한 테마를 구분합니다.
화면 밖으로 잘린 수업의 종료 시각과 누락 글자는 추측하지 않습니다.
과목명/강의실 일치율, 요일 및 시작·종료 시각 정확도, 블록 누락/중복,
최초·캐시 후 처리 시간과 최대 메모리, 다운로드 크기를 평가합니다.
원본과 동일한 중복 이미지는 평가에서 중복 집계하지 않습니다.
사용자 스크린샷은 현재 저장소에 업로드하지 않았습니다.

## 참고 자료

- https://github.com/naptha/tesseract.js
- https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_recognition.html
- https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/cross_platform/browser.html
- https://onnxruntime.ai/docs/tutorials/web/

아직 샘플 OCR 정확도, 처리시간, 최종 배포 크기를 측정하지 않았습니다.
