# Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared
 This paper has been accpeted by CVPR 2026 Highlight [[arxiv](https://arxiv.org/abs/2603.08018)]
 
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

- **Qwen-7B-Chat**: [Hugging Face Link](https://huggingface.co/Qwen/Qwen-7B-Chat)

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

All the model weights will be uploaded successively to [Link](https://drive.google.com/drive/folders/154jeD1NgNDXg8rEMSzNg4He5Vd9-Uc5q?usp=sharing)
