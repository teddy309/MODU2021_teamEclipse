# MODU2021_teamEclipse

clone git repository </br>
- git clone https://github.com/teddy309/MODU2021_teamEclipse 

conda 환경 install (아래 중 하나) </br>
- conda install --file requirements.txt (updated)
- conda env create --file environment.yaml

```python
  ##conda 가상환경 관련	
  (python3:venv) [출처: https://hcnoh.github.io/2018-10-06-ubuntu-python-virtualenv]
  - 가상환경 생성: (디렉토리 이동 후)virtualenv -p python3 myVenv
  - 활성화: (상대경로)myVenv\Scripts\acativate 파일경로 입력 (비활: deactivate)
  (conda)		
  - 확인: conda info --envs
  - conda환경 이름바꾸기: 복제하고 기존환경은 삭제해야됨....
    $ conda create --name [변경할이름] --clone [기존환경이름]
    $ conda activate [변경할이름]
    $ conda remove --name [기존환경이름] --all

  출처: https://gentlesark.tistory.com/19 [삵 izz well]
  - 환경 생성/복제: conda create -n [생성할venv] --clone [복제할venv]
                    conda env create -f environment.yml && conda activate [yml conda_name]
  - 환경 생성 및 python버전 설정(3.6): conda create --name [venv_py36] python=3.6
  - 환경 삭제: conda env remove -n [삭제할venv]
  - 시작/종료: conda activate [lssVenv1] / conda deactivate
  - 가상환경export: conda list --export > packagelist.txt 
            import: conda install --file packagelist.txt 
  (jupyter notebook/lab)
  주피터 커널:	- 생성: python -m ipykernel install --user --name 가상환경 이름 --display-name 커널 이름
  - 목록: jupyter kernelspec list
  - 삭제:
  - 커널 삭제: jupyter kernelspec uninstall 커널 이름
  - 가상환경 삭제: conda remove --name 가상환경 이름 --all
  - 수정:

```

# python pipeline 파일 실행
- conda activate [가상환경 이름] 
- bash run_tasks.sh : 파이썬 파일 순차실행 [pipeline_COLA.py pipeline_COPA.py pipeline_BoolQ.py]


- python format_check.py (국립국어원 제출형식 포맷체크 파일)

# updated
- 7/21 : 불필요한 파일 삭제 및 BoolQ/WiC 수정완료.


# TODO
* pipeline_WiC.py on updating -> WiC/BoolQ pipeline updated(**task에 맞게 변형해서 사용하기.**)
- [옵션] add argparse, tensorboard
- **일부폴더 삭제 예정** (datasets/* 8/1에 삭제 예정) 
