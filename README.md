# :raised_hand: [Boostcamp-AI-Tech-Level3] HiBoostCamp :raised_hand:

:notes:

## 최종 프로젝트 목표
피아노 연주영상에서 악보(MIDI)생성

## 개발 배경 및 필요성
- 오디오 -> 악보 기술의 성능 한계 극복 (노이즈가 들어갈 경우 성능의 급격한 저하 발생)
- 시각 데이터를 사용하여 영상에서 손실/왜곡된 음향 데이터 복원/생성
- 확장가능성<br>
  입력 피아노 영상에서 생성한 Midi 파일을 사용하여 다른 악기로 연주한 버전 생성

## Baseline 모델
- _Su, Kun, Xiulong Liu, and Eli Shlizerman. "Audeo: Audio generation for a silent performance video." Advances in Neural Information Processing Systems 33 (2020): 3325-3337._
- _Gan, Chuang, et al. "Foley music: Learning to generate music from videos." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16. Springer International Publishing, 2020._

## 사용 중인 데이터 셋
- <a href="http://data.csail.mit.edu/clevrer/data_pose_midi.tar">Foley Music 논문에서 사용한 데이터셋</a>
- <a href="https://github.com/shlizee/Audeo">Audeo 논문에서 사용한 데이터셋</a>
- <a href="https://www.robots.ox.ac.uk/~vgg/research/sighttosound/">PianoYT 데이터셋</a>

## 역할 분담
| 역할 | 담당 개발자 |
| :-- | :--------- |
|모델성능개선|이종목, 정성혜|
|데이터 수집, relabeling|이종목|
|데이터 전처리|강나훈|
|streamlit, gradio 서빙 (frontend / backend)|김근욱|
|CI/CD 프로젝트 관리 (Git Action) 및 시각화 조사|김희상|

## Contributors?
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
[![Version](https://img.shields.io/badge/Version-0.1-green.svg?style=flat-square)](#version-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/ejrtks1020"><img src="https://github.com/ejrtks1020.png" width="100px;" alt=""/><br /><sub><b>강나훈</b></sub></a><br /><a href="https://github.com/ejrtks1020" title="Code"></td>
    <td align="center"><a href="https://github.com/lijm1358"><img src="https://github.com/lijm1358.png" width="100px;" alt=""/><br /><sub><b>이종목</b></sub></a><br /><a href="https://github.com/lijm1358" title="Code"></td>
    <td align="center"><a href="https://github.com/fneaplle"><img src="https://github.com/fneaplle.png" width="100px;" alt=""/><br /><sub><b>김희상</b></sub></a><br /><a href="https://github.com/fneaplle" title="Code"></td>
    <td align="center"><a href="https://github.com/KimGeunUk"><img src="https://github.com/KimGeunUk.png" width="100px;" alt=""/><br /><sub><b>김근욱</b></sub></a><br /><a href="https://github.com/KimGeunUk" title="Code"></td>
    <td align="center"><a href="https://github.com/jshye"><img src="https://github.com/jshye.png" width="100px;" alt=""/><br /><sub><b>정성혜</b></sub></a><br /><a href="https://github.com/jshye" title="Code"></td>    
  </tr>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
