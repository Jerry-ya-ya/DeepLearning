py -3.11 -m venv .venv

.\.venv\Scripts\activate

python -m pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/WongKinYiu/yolov7.git

cd yolov7

pip install -r requirements.txt

Execute check_gpu.py

https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions

Step...