# Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared
 This paper has been accpeted by CVPR 2026 Highlight [[arxiv](https://arxiv.org/abs/2603.08018)] [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Missing_No_More_Dictionary-Guided_Cross-Modal_Image_Fusion_under_Missing_Infrared_CVPR_2026_paper.html)] 
 
 <div align=center>
<img src="https://github.com/harukiv/DCMIF/blob/main/overview.png" width="90%">
</div>

##  Environment
- Python >= 3.8
- PyTorch == 2.6.0 is recommended
- opencv-python == 3.4.9.31
- tqdm

##  Dependencies
This project uses the following pretrained LLM:

- **Qwen-7B-Chat**: [[BaiduPan]]( https://pan.baidu.com/s/14re0TNK4scsXs1Dj1zqrwQ?pwd=yrvj) (password: yrvj)

Please download the model from the official page and place it in:

```bash
./LLM/Qwen
```

## Train
The model is trained in three stages: JSRL, VGII and AFRI. 

First, the shared dictionary and the weights of the coefficient encoding and decoding network used in the subsequent stage are obtained through the JSRL training. Among them, the weight of the dictionary is saved as **dictionary.pth**. Subsequently, we continued to train the VGII module, obtaining the corresponding weights for the pseudo-infrared inference network. Finally, the AFRI module is re-trained to obtain the fusion result.

##  Test
Just run the **"AFRI_test.py"** file. The weights that need to be loaded include the training weights of AFRI and the dictionary weights **"dictionary.pth"**.

The model was trained on three datasets, and the fusion weights were different on each dataset: MSRS, FLIR, and KAIST.

All the model weights will be uploaded successively to [Google Drive Link](https://drive.google.com/drive/folders/154jeD1NgNDXg8rEMSzNg4He5Vd9-Uc5q?usp=sharing)

## Citation
```
@InProceedings{Zhang_2026_CVPR,
    author    = {Zhang, Yafei and Ma, Meng and Li, Huafeng and Liu, Yu},
    title     = {Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {19549-19558}
}
```
