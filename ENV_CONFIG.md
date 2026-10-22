# SGLang on Ascend 910b3, running Qwen

## reference web page

https://github.com/sgl-project/sglang/blob/v0.5.20/docs/docs/hardware-platforms/ascend-npus/getting-started/installation.mdx
current main docs has already remove contents about 910b, so using docs on sglang tag v0.5.20.

## conda env create

```bash
conda create --name sglang_npu python=3.11
conda activate sglang_npu
```

## CANN configuration

### install NPU driver and firmware

```bash
# need root
```

### install CANN toolkit

```bash
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-aarch64.run
mkdir -p /data/zkw/Ascend
mkdir -p /data/zkw/tmp
export TMPDIR=/data/zkw/tmp
bash ./Ascend-cann-toolkit_9.0.0_linux-aarch64.run --install --install-path=/data/zkw/Ascend
echo 'source /data/zkw/Ascend/cann/set_env.sh' >> ~/.bashrc
```

### install CANN ops

```bash
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-910b-ops_9.0.0_linux-aarch64.run
mkdir -p /data/zkw/tmp
export TMPDIR=/data/zkw/tmp
bash ./Ascend-cann-910b-ops_9.0.0_linux-aarch64.run --install --install-path=/data/zkw/Ascend
```

### install NNAL (optional)

```bash
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-nnal_9.0.0_linux-aarch64.run
mkdir -p /data/zkw/tmp
export TMPDIR=/data/zkw/tmp
bash ./Ascend-cann-nnal_9.0.0_linux-aarch64.run  --install --install-path=/data/zkw/Ascend
echo '# source /data/zkw/Ascend/nnal/atb/set_env.sh' >> ~/.bashrc
echo '# source /data/zkw/Ascend/nnal/asdsip/set_env.sh' >> ~/.bashrc
```

### configure after install

```bash
pip install attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20.0 scipy requests absl-py
```

### validate after install

```bash
# expected out: Ascend910B3
python -c "import acl;print(acl.get_soc_name())"
```

## MemFabric-Hybrid

If you want to use PD disaggregation mode, you need to install MemFabric-Hybrid. MemFabric-Hybrid is a drop-in replacement of Mooncake Transfer Engine that enables KV cache transfer on Ascend NPU clusters.

```bash
pip install memfabric-hybrid==1.0.8
```

## MemFabric-zbal

MemFabric-zbal is a Zero Buffer Acceleration Library of high-performance operators for LLM inference and training on Ascend, accelerating computation by eliminating intermediate memory buffers; it is required only on aarch64 clusters and is installed in addition to MemFabric-Hybrid.

```bash
# Only needed on aarch64 (arm64) hosts
pip install memfabric-zbal==1.1.1
```

## PyTorch and PyTorch Framework Adapter on Ascend

```bash
PYTORCH_VERSION=2.10.0
TORCHVISION_VERSION=0.25.0
TORCH_NPU_VERSION=2.10.0
pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==$TORCH_NPU_VERSION
```

## Triton on Ascend

```bash
pip install triton-ascend==3.2.1.dev20260530 \
  --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly \
  --trusted-host mirrors.huaweicloud.com
```

### sglang kernel npu

```bash
# clone project
cd Ascend/
git clone https://github.com/sgl-project/sgl-kernel-npu.git
cd sgl-kernel-npu
git submodule update --init --recursive

# SGLang Kernels for Ascend NPU
# important: make some change in CMakeLists.txt since no access to drivers in /usr
source /home/zkw/data/Ascend/ascend-toolkit/set_env.sh
bash build.sh -a kernels
pip install output/sgl_kernel_npu*.whl
# (Optional) Confirm whether the import can be successfully
python -c "import sgl_kernel_npu; print(sgl_kernel_npu.__path__)"
source /home/zkw/data/Ascend/ascend-toolkit/set_env.sh

# DeepEP-Ascend
source /home/zkw/data/Ascend/ascend-toolkit/set_env.sh
bash build.sh -a deepep
pip install output/deep_ep*.whl
# Link to the deep_ep_cpp.*.so file
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so && cd -
# (Optional) Confirm whether the import can be successful
python -c "import deep_ep; print(deep_ep.__path__)"
source /home/zkw/data/Ascend/ascend-toolkit/set_env.sh

## back to Ascend dir
cd /data/zkw/Ascend
```

## Some other dependencies

```bash
# libGL
apt update
apt install libgl1 libglib2.0-0

# ensure setuptools contains pkg_resources module
pip install "setuptools<80"
```

## Installing SGLang from source

```bash
# Use the last release branch
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout tags/v0.5.2
mv python/pyproject_npu.toml python/pyproject.toml
pip install -e python[all_npu]
```

## model download

```bash
# choose Qwen/Qwen3-32B and Qwen/Qwen3-30B-A3B-Instruct-2507 since they're supported.
conda create -n hf python=3.11 pip -y
conda activate hf
python -m pip install -U huggingface_hub
conda env config vars set -n hf \
  HF_HOME=/home/zkw/data/hf_home \
  HF_HUB_CACHE=/home/zkw/data/hf_home/hub \
  HF_XET_CACHE=/home/zkw/data/hf_home/xet
conda deactivate
conda activate hf
mkdir -p /home/zkw/data/models
mkdir -p /home/zkw/data/hf_home
hf auth login
hf download Qwen/Qwen3-32B \
  --local-dir /home/zkw/data/models/Qwen3-32B
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir /home/zkw/data/models/Qwen3-30B-A3B-Instruct-2507
```

## model running

```bash
# local.bash: start model with local env
export ASCEND_RT_VISIBLE_DEVICES=2,3,4,5
source /home/zkw/data/Ascend/ascend-toolkit/set_env.sh
python -m sglang.launch_server \
    --model-path /home/zkw/data/models/Qwen3-32B \
    --tp-size 4 \
    --device npu \
    --attention-backend ascend \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.8

# test.bash: start model with container
# ... (see the file)
```
