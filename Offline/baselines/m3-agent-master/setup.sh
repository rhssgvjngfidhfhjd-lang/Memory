pip install -r requirements.txt

sudo pip install setuptools_scm torchdiffeq resampy x_transformers
pip install accelerate==0.34.2 # https://github.com/huggingface/trl/issues/2377
sudo pip install ninja
sudo apt-get -y install ninja-build

pip install flash-attn==2.6.3 --no-build-isolation
sudo pip install torch==2.6.0
sudo pip install torchvision==0.21.0
sudo pip install torchaudio==2.6.0
pip install peft==0.15.2
pip install moviepy==2.1.2
sudo pip install jupyter
sudo pip install httpx==0.23.0
pip install pydub==0.25.1

sudo pip install trl==0.16.0 # other versions may have problems
sudo apt-get -y install ffmpeg # load audio in video(mp4)

pip install glob
pip install onnxruntime==1.22.1
pip install insightface==0.7.3
pip install hdbscan==0.8.40
