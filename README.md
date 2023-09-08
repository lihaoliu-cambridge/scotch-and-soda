# SCOTCH and SODA: A Transformer Video Shadow Detection Framework

This repository provides a Pytorch implementation of the paper ["SCOTCH and SODA: A Transformer Video Shadow Detection Framework, CVPR'23"](https://arxiv.org/abs/2211.06885).

<img src="https://github.com/lihaoliu-cambridge/lihaoliu-cambridge.github.io/blob/master/scotch_and_soda/figures/segmentation_results.png">  

## Requirement

cuda==11.1   
cudnn==8.0  
torch==1.9.0   
timm==0.9.6   
transformers==4.30.2   
pytorch-lightning==1.5.10  
medpy==0.4.0  
einops==0.6.1


## Usage Instructions

1. **Setup**

   Clone the repository and navigate to its directory:

   ```shell
   git clone https://github.com/lihaoliu-cambridge/scotch-and-soda.git
   cd scotch-and-soda
   ```
   
2. Dataset Preparation

   Download and unzip [ViSha dataset](https://erasernut.github.io/ViSha.html), Place the unzipped Visha directory into the dataset directory:
   
   ```shell
   ./dataset/ViSha
   ```
   
3. Configuration
   
   Adjust the configurations for the dataloader, model architecture, and training logic in:
         
   ```shell
   ./config/scotch_and_soda_visha_image_config.yaml
   ```
   
4. Training
   
   To train the model, execute:
    
   ```shell
   python train.py
   ```

   **Note**: Due to the large GPU memory requirement from the video-level dataloader, the dataloader has been switched to an image-level dataloader for easy training, which gives comparable results to the video-level dataloader. It's also advised to first train with the image-level dataloader and subsequently fine-tune with the video-level dataloader for fast convergency.

5. Monitoring with Tensorboard

   To view the training progress, start Tensorboard and open http://127.0.0.1:6006/ in your browser:
   
   ```shell
   tensorboard --port=6006  --logdir=[Your Project Directory]/output/tensorboard/scotch_and_soda_visha_image
   ```
   <img src="https://github.com/lihaoliu-cambridge/lihaoliu-cambridge.github.io/blob/master/pic/papers/vsd_visualization.png" width="960"/>  

6. Testing

   After training, update the checkpoint file path in the test.py script. Then, test the trained model using:
   
   ```shell
   python test.py
   ```

## Results from Scotch and Soda

We have evaluated our "Scotch and Soda" model on the ViSha testing set. The results have been made available for viewing and download on [Google Drive](https://drive.google.com/drive/folders/11as6nfNav6aBEMzlK3H9QnuV0NyIRJV3?usp=sharing).

## Citation

If you use this code or the associated paper in your work, please cite:
   
```
@inproceedings{liu2023scotch,
   title={SCOTCH and SODA: A Transformer Video Shadow Detection Framework},
   author={Liu, Lihao and Prost, Jean and Zhu, Lei and Papadakis, Nicolas and Li{\`o}, Pietro and Sch{\"o}nlieb, Carola-Bibiane and Aviles-Rivero, Angelica I},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   pages={10449--10458},
   year={2023}
}
```
