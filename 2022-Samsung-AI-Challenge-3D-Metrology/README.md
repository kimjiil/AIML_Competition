## Experiment List

블로그 내용 정리
[https://kimjiil.github.io/pytorch%20study/dacon-study/](https://kimjiil.github.io/pytorch%20study/dacon-study/)

### 1. 221202_AE 
대회에서 주어진 basic AutoEncoder로 학습
공유된 코드로 재현 및 데이터 분석

[jupyter notebook](./221202_AE/221202_basic_AE.ipynb)

<p align="center">
<img src="/2022-Samsung-AI-Challenge-3D-Metrology/221202_AE/AE_summit1.PNG"
height="100%" width="100%">
</p>


### 2. 221209_cyclegan
cycle gan으로 학습했으나 실패함(학습기의 구조적인 문제?)    

- [jupyter notebook](./221209_cyclegan/221209_cyclegan(fail).ipynb)

### 3. 221216_cyclegan
221209에서 네트워크에 들어가는 input이 잘못 들어가 학습이 안되는 오류수정


- [jupyter notebook](./221216_cyclegan/221216_cyclegan_semtodepth.ipynb)
  - Simulation SEM Image에서 Simulation Depth Image로 변환하는 cycle Gan을 학습한 코드

- [jupyter notebook](./221216_cyclegan/221216_cyclegan_simtotrain.ipynb)
  - Simulation SEM Image에서 Real SEM Image로 변환하는 cycle Gan을 학습한 코드

[[실험 결과 링크]](https://wandb.ai/kimjiil2013/Samsung%20sem%20CycleGan%20221216/table?workspace=user-kimjiil2013)