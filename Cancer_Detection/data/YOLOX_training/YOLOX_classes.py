"""
This is for the dependencies of the yolox
"""
KAGGLE = True

!git clone https://github.com/roboflow-ai/YOLOX.git %cd YOLOX
!pip3 install -U pip && pip3 install -r requirements.txt
!pip3 install -v -e . 
!pip uninstall -y torch torchvision torchaudio # May need to change in the future if Colab no longer uses CUDA 11.0 !pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
%cd /content/
!git clone https://github.com/NVIDIA/apex
%cd apex
!pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
!pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

