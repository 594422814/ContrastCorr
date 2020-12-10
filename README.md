# Contrastive Transformation for Self-supervised Correspondence Learning

Ning Wang, Wengang Zhou, and Houqiang Li 

To appear in *AAAI 2021*

![](../main/ContrastCorr.png)

## Prerequisites
The code is tested in the following environment:
- Ubuntu 16.04
- Pytorch 1.1.0, [tqdm](https://github.com/tqdm/tqdm), scipy 1.2.1

## Training on the TrackingNet

### Dataset

We use the [TrackingNet dataset](https://tracking-net.org/) for model training.

### Training command

```
python train_trackingnet.py 
```

## Testing on DAVIS2017
To test on DAVIS2017 for instance segmentation mask propagation, please run:
```
python test.py -d /workspace/DAVIS/ -s 560
```
Important parameters:
- `-c`: checkpoint path.
- `-o`: results path.
- `-d`: DAVIS 2017 dataset path.
- `-s`: test resolution, all results in the paper are tested on 560p images, i.e. `-s 560`.

Please check the `test.py` file for other parameters.

## Testing on the VIP dataset

To test on VIP, please run the following command with your own VIP path:

```
python test_mask_vip.py -o results/VIP/category/ --scale_size 560 560 --pre_num 1 -d /DATA/VIP/VIP_Fine/Images/ --val_txt /DATA/VIP/VIP_Fine/lists/val_videos.txt -c weights/checkpoint_latest.pth.tar
```
and then:
```
python eval_vip.py -g DATA/VIP/VIP_Fine/Annotations/Category_ids/ -p results/VIP/category/
````

## Citation
If you find this work useful for your research, please consider citing our work:

```
@inproceedings{Wang_2021_Contrastive,
    title={Contrastive Transformation for Self-supervised Correspondence Learning},
    author={Wang, Ning and Zhou, Wengang and Li, Houqiang},
    booktitle={AAAI},
    year={2021}
}
```

## Acknowledgements
- This code is based on the [UVC](https://github.com/Liusifei/UVC) and [TimeCycle](https://github.com/xiaolonw/TimeCycle).
