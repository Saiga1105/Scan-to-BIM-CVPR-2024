# T1. Semantic Segmentation


## steps to get PTv3 running on remote Linux

- clone repo on linux machine


- create environment
```bash
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# We use CUDA 11.8 and PyTorch 2.1.0 for our development of PTv3
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

cd libs/pointops
python setup.py install
cd ../..

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu118  # choose version match your local cuda version

# Open3D (visualization, optional)
pip install open3d
```
- install Flash Attention
    - pip install flash-attn --no-build-isolation

- clone https://github.com/Saiga1105/Scan-to-BIM-CVPR-2024.git to linux server

- To incorporate this code into your project, clone the project repo and copy the following file/folder to your project:
```bash
PATH_TO_YOUR_PROJECT='/home/mbassier/code/Scan-to-BIM-CVPR-2024/scripts'
cp model.py ${PATH_TO_YOUR_PROJECT} 
cp -r serialization ${PATH_TO_YOUR_PROJECT}
```


## steps to get PTv3 running on Windows WSL

- install WSL
- install Conda
    - wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh # get installer script
    - chmod +x Miniconda3-latest-Linux-x86_64.sh #give permissions
    - ./Miniconda3-latest-Linux-x86_64.sh # run installer script
    - source ~/.bashrc #activate conda
    - conda list #verify installation

- create environment
```bash
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# We use CUDA 11.8 and PyTorch 2.1.0 for our development of PTv3
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

cd libs/pointops
python setup.py install
cd ../..

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu118  # choose version match your local cuda version

# Open3D (visualization, optional)
pip install open3d
```

- Install NVIDIA GPU Drivers for Windows (Optional):
    - Ensure that you have the latest NVIDIA GPU drivers installed that support WSL 2. These drivers are different from the standard drivers and are often labeled as DCH drivers that include support for CUDA on WSL.

- Download and install the CUDA Toolkit for WSL, available from NVIDIA's official site. This toolkit typically includes support for CUDA, cuDNN, and TensorRT under WSL 2.
    - https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run
```

- Install the CUDA Software
    ```bash
    conda install cuda -c nvidia
    ```

## steps to get PTv3 running on Windows
- Download and install the CUDA Toolkit
    - https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local

- Install the CUDA Software
```bash
conda install cuda -c nvidia
```