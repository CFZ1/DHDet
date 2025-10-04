# Dual-head detector with point-driven transformer and semantic-spatial gating for liquid crystal display defects

The official PyTorch implementation of Dual-head detector with point-driven transformer and semantic-spatial gating for liquid crystal display defects. This project is based on PyTorch, MMDetection 3.x, and MMYOLO. It is worth noting that MMDetection 3.x has significant architectural changes compared to MMDetection 2.x.

## Prerequisites

We have tested in a [**Python=3.8**](https://www.python.org/) environment with [**PyTorch=1.10.0**](https://pytorch.org/get-started/previous-versions/). Other environments may work as well. 

```python
conda create --name DHDet python=3.8
conda activate DHDet
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
# download mmdetection
cd /root/DHDet_codes/mmdetection #3.3.0
pip install -e . 
cd /root/DHDet_codes/DHDet
pip install -e .

# others
pip install imgaug==0.4.0
pip install tqdm==4.61.2
pip install similaritymeasures==1.1.0
```

## Dataset

 - `the PVEL-AD dataset`: The PVEL-AD dataset is a public defect dataset. It contains images of photovoltaic solar cells. You can download it from [here](https://github.com/binyisu/PVEL-AD). 

 - `the LCD-DET and MSD-DET datasets`: These two datasets are self-built defect datasets. They contain images of LCD panels and glass mobile screens, respectively. You can access 10 LCD image samples with their corresponding annotations (approximately 40MB) through the following links:

   - [Baidu Netdisk](https://pan.baidu.com/s/1LdOR9JCB8CCb_IoD8cA8hA?pwd=3bxr) (Extraction code: 3bxr)

   - [Google Drive](https://drive.google.com/file/d/1LaOrHZeql5Q59daweJzGHs5xyDoYFMJr/view?usp=drive_link)

 - The annotations are created using the [X-AnyLabeling software](https://github.com/CVHub520/X-AnyLabeling). 

## Code Structures

The code structure follows a similar organization to MMDetection 3.x and MMYOLO. It mainly consists of five parts:

 - `tools`：Contains scripts for training and testing processes.
 - `configs` or `0_configs`: Contains configuration files for experiments.
 - `mmyolo/datasets`: Includes dataloaders for different datasets and data augmentation methods.
 - `mmyolo/models`: Contains various model components and architectures.
 - `mmyolo/evaluation`: Contains evaluation metrics and assessment tools.

## Training scripts

4 gpus

```python
conda activate DHDet
cd /root/DHDet_codes/DHDet/0_configs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ../tools/train.py ./DHDet_LCDDET.py --launcher pytorch
```

single gpu

```python
conda activate DHDet
cd /root/DHDet_codes/DHDet/0_configs
CUDA_VISIBLE_DEVICES=0 python ../tools/train.py ./DHDet_LCDDET.py
```

## Test scripts

```python
conda activate DHDet
cd /root/DHDet_codes/DHDet/0_configs
python ../tools/test.py ./DHDet_LCDDET.py /root/epoch_60.pth --out ./DHDet_LCDDET_epoch_60_test.pkl
```

## Acknowledgment

We thank the following repos providing helpful components/functions/dataset in our work.

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMYOLO](https://github.com/open-mmlab/mmyolo)
- [the PVEL-AD dataset](https://github.com/binyisu/PVEL-AD)

