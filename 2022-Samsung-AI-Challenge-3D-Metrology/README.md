## 2022 Samsung AI Challenge 3D Metrology

### 블로그 내용 정리
[https://kimjiil.github.io/pytorch%20study/dacon-study/](https://kimjiil.github.io/pytorch%20study/dacon-study/)

### 1. 221202_AE 
- 대회에서 주어진 basic AutoEncoder로 학습
- 공유된 코드로 재현 및 데이터 분석

[[jupyter notebook]](./221202_AE/221202_basic_AE.ipynb)

<p align="center">
<img src="/2022-Samsung-AI-Challenge-3D-Metrology/221202_AE/AE_summit1.PNG"
height="100%" width="100%">
</p>


### 2. 221209_cyclegan
- cycle gan으로 학습했으나 실패함(학습기의 구조적인 문제?)    

- [[jupyter notebook]](./221209_cyclegan/221209_cyclegan(fail).ipynb)

### 3. 221216_cyclegan
- 221209에서 네트워크에 들어가는 input이 잘못 들어가 학습이 안되는 오류수정

- [[jupyter notebook]](./221216_cyclegan/221216_cyclegan_semtodepth.ipynb)
  - Simulation SEM Image에서 Simulation Depth Image로 변환하는 cycle Gan을 학습한 코드

- [[jupyter notebook]](./221216_cyclegan/221216_cyclegan_simtotrain.ipynb)
  - Simulation SEM Image에서 Real SEM Image로 변환하는 cycle Gan을 학습한 코드

- testset에서 case간의 depth 분포차이

<p align="center">
<img src="/2022-Samsung-AI-Challenge-3D-Metrology/221216_cyclegan/semtodepth validation result.png"
height="50%" width="50%">
</p>

- [[wandb.ai 실험결과 링크]](https://wandb.ai/kimjiil2013/Samsung%20sem%20CycleGan%20221216/table?workspace=user-kimjiil2013)

### 4. 221220_cyclegan_cnncls
- 221216에서 전체 case에 대해 cycle gan을 학습시킨 결과 case간의 어느정도 분포가 구별되지만 만족할만한 성능은 나오지 않는것 같아서
case별로 나누어서 cycle gan을 학습시킴.

- SEM Image의 case를 분류하기 위한 CNN classifier를 추가함.

- [[jupyter notebook]](./221220_cyclegan_cnncls/221220_cnn_classifier_samsung.ipynb)
  - SEM Image의 case를 분류하기 위해 cnn classifier를 학습한 코드
  - [[wandb.ai 실험결과 링크]](https://wandb.ai/kimjiil2013/Samsung%20sem%20CycleGan/runs/1sn5oje3?workspace=user-kimjiil2013)

- [[jupyter notebook]](./221220_cyclegan_cnncls/221220_cyclegan_cnncls.py.ipynb)
  - case별로 cycle gan을 나누어서 학습한 코드
  
[[wandb.ai 실험 결과 링크]](https://wandb.ai/kimjiil2013/Samsung%20sem%20CycleGan?workspace=user-kimjiil2013)

### 5. 221227_cyclegan_cnncls

- 221220에서 성능을 좀더 올리기 위해 paired dataset인 Simulation SEM/Depth Image간의 Guided L1 Loss를 추가하여 
depth를 더 잘 학습하도록 함.

- CNN classifier는 221220_cyclegan_cnncls에서 학습한 모델 사용.

- [[jupyter notebook]](./221227_cyclegan_cnncls/jupyter%20notebook/221227_cyclegan_train.ipynb)
  - case별로 학습, submission data를 cnn classifier로 case 분류 하고 cycleGan model로 Depth Image 생성

[[221227 wandb 실험 결과 링크]](https://wandb.ai/kimjiil2013/221227_Samsung_sem?workspace=user-kimjiil2013)

- 대회 기간은 지나 갔지만 제출한 데이터는 139명중 30등보다 높게 성능이 나옴.

<p align="center">
<img src="/2022-Samsung-AI-Challenge-3D-Metrology/221227_cyclegan_cnncls/cyclegan_result.PNG"
height="100%" width="100%">
</p>

<p align="center">
<img src="/2022-Samsung-AI-Challenge-3D-Metrology/221227_cyclegan_cnncls/leader_board_chart.PNG"
height="100%" width="100%">
</p>






