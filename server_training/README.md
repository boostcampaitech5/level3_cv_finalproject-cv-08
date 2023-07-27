# Music Transcription from Silent Videos - Training server
## Python scripts
* `Video2Roll_train.py` : Video2Roll 훈련
* `Video2Roll_inference.py` : Video2Roll inference
* `Video2Roll_evaluate.py` : Video2Roll metric 계산
* `Roll2Midi_train.py` : Roll2Midi 훈련
* `Roll2Midi_inference.py` : Roll2Midi inference
* `Roll2Midi_evaluate.py` : Roll2Midi metric 계산

## `dataset/`
Video2Roll dataset, augmentation, class balanced sampler 정의

## `model/`
Video2Roll Network 및 backbone 변형 구현체 정의
  
## `notebooks/`
실험 및 각종 도구 jupyter notebook
* `EDA.ipynb` : 데이터 EDA
* `experiment.ipynb` : 짧은 실험 및 검증을 위한 notebook
* `get_videos.ipynb` : 특정 유튜버의 영상 link 크롤링
* `visualization.ipynb` : 데이터 및 inference 결과 시각화

## `tools/`
데이터셋 제작을 위한 tool
* `audeo_makelabels.py` : midi label을 audeo용 label로 변환
* `audio2flac.py` : 음성 파일을 flac 확장자로 변환(Onsets and Frames 전용)
* `join_traindata.py` : 여러 directory의 training data들을 하나의 directory로 link
* `video2images.py` : 영상을 프레임 별로 crop, resize 후 저장
* `yt_download.py` : 링크로부터 유튜브 영상 다운로드

## `trainer/`
Video2Roll trainer 저장

## `utils/`
config validator등 utility 파일 저장