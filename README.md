# SVM and RSF Pipeline for Clinical Data Analysis

![WSI-QA](SVM-RSF.png)

<p align="justify"> This pipeline provides a general framework for analyzing clinical data using both classification and survival analysis models. It begins with constructing a feature matrix, where each row corresponds to a patient and each column represents a clinical variable. Missing values are addressed through imputation, and features are normalized to ensure comparability. The processed data is then used to train two types of models: a classification model-Support Vector Machine, and a survival model-Random Survival Forest, to estimate individual survival probabilities over time. Model performance is evaluated using 5-fold cross-validation, ensuring reliability and generalizability across subsets of the data. This flexible pipeline supports a wide range of clinical prediction tasks, from risk group classification to time-to-event forecasting. </p>

# Setting Up the Pipeline:

1. System requirements:
- Ubuntu, Windows, Mac
- Python version >= 3.9 (using conda environments)
- Anaconda version >= 23.7.4
2. Steps to Set Up the Pipeline:
- Download the pipeline to your Desktop
- Navigate to the downloaded pipeline folder
- Open Terminal then navigate to pipeline folder  `cd path/to/pipeline/folder`
- Create a conda environment:
`conda create -n SVM-RSF python=3.9`

- Activate the environment:
  `conda activate SVM-RSF`

- Install required packages:
  `pip install -r requirements.txt`


# 1. Dataset:

<p align="justify"> Prepare your dataset in Excel files and place them in the `/data` folder — one file for the SVM model and one for the RSF model. For the SVM file, the first column should contain patient IDs, followed by clinical variables as columns, and the last column must contain the classification label (e.g., 0: negative, 1: positive). For the RSF file, the structure is similar: the first column is the patient ID, followed by clinical variables, with the last two columns as "time" and "event" (e.g., [Survival_month, Survival_status]). Example files for both formats are provided in the `/data` folder.</p>

# 2. Preprocessing:

The preprocessing consists of the following steps:
- Data Imputation - Missing values are handled using two strategies:
  - MICE (Multiple Imputation by Chained Equations): Estimates missing values based on relationships between variables.
  - Constant Imputation: Replaces missing values with a fixed value (e.g., 0 or a specific value)
- Data Normalization:
After imputation, all features are normalized to ensure they are on a consistent scale. This prevents variables with larger ranges from dominating the learning process and helps improve model training and performance. Two normalization strategies are used:
  - Z-score scaling: Transforms features to have zero mean and unit variance, useful for models that assume Gaussian-like distributions.
  - Min-max scaling: Rescales features to a fixed range, typically [0, 1], preserving relative relationships and improving performance for distance-based models.


To preprocess the data for SVM, run the following command:
```bash
python pre-processing.py \
  --input_dir data/SVM_data.xlsx \
  --output_dir output \
  --features_to_normalize Sex CTPs Nodule_No Multinodular_tumor BCLC Age BMI MTD ALT AST ALP ALB Bili WCC Hb Neu Lym PLT CRP AFP ALBIs ALBIg \
  --scaling_method MinMaxScaler \
  --imputation_method MICE \
  --replace_val -1 \
  --add_indicator False \
  --do_norm True
```
Notes: 
- When using `MinMaxScaler`, you need to manually specify only the features that require normalization, as some features may already be in the [0, 1] range and re-scaling them could distort their values. In contrast, when using Z-score normalization (StandardScaler), all features should be normalized regardless of their original range, so you should provide the full list of feature names.
- Choose the scaling method by setting `--scaling_method` to either `MinMaxScaler` or `StandardScaler`.
- Select the imputation method by setting `--imputation_method` to either `MICE` or `const`.
- Set `--replace_val` to specify a constant value for replacing missing entries (default is -1).
- Add a missing-value indicator column by setting `--add_indicator` to True.
- Enable feature normalization by setting `--do_norm` to True; set to False to skip normalization.

To preprocess the data for RSF, run the same command as before but change the input file to `--input_dir data/RSF_data.xlsx` and set `--do_norm` to False. Normalization is not required for RSF.

# 3. SVM:

After preprocessing your dataset, you can train and evaluate the Support Vector Machine (SVM) classifier using the `SVM.py` script. This script accepts several configurable parameters for tuning and experimentation.
```bash

python SVM.py \
  --dataset output/feature_filled_norm.xlsx \
  --output_dir output \
  --kernel_fn rbf \
  --C 1.0 \
  --gamma 0.55
```
Notes:
- Set `--dataset` to the path of the preprocessed Excel file you want to use for training and evaluation (e.g., output/feature_filled_norm.xlsx).
- Set `--output_dir` to define where the model’s results will be saved (defalut is `output`)
- Choose the SVM kernel by setting `--kernel_fn` to one of the following options: linear, rbf, poly, or sigmoid (default is rbf).
- Control the regularization strength using `--C (default is 1.0).
- Set `--gamma` to define the kernel coefficient. Acceptable values include 'scale', 'auto', or a float ( default is 0.55).



# 4. RSF:






you need to privide the name of the features that need normalization in the case of min-max. some features may be alreday in the range of 0-1, so no need to normalize. 

After imputation, all features are normalized to ensure a consistent scale across variables. This helps prevent features with larger ranges from dominating the learning process and improves model convergence and performance.



<p align="justify"> prepare your dataset in exel files and put them in folder `/data`, one for SVM and one for RSF. the SVM file the rist coulm is pairtnt ID,  in the row and clincal varalbes in the coumn, the last colum must be the classification label (eg., 0:postive, 1: negative). SImilar for RSF, first coulm paient ID foolowed by clinical varbles, and the last two columnss are "time" and "event" (eg., [Survival_month,Survival_status]). two examples are provided in folder  `/data`

OS_m	Survival_status ![image](https://github.com/user-attachments/assets/6b6b643d-e3c0-43e0-919b-7f0db5114ea0)

  
  
  
  The pipeline starts by identifying the WSI tissue region and dividing it into smaller image tiles (e.g., 270x270). Pen-marking detection is then applied to categorize the tiles into two classes: those with high pen-marking (which are discarded) and those with medium and low pen-marking. Tiles with medium and low pen-marking undergo a pen-marking removal process, resulting in clean image tiles. Next, the clean image tiles are fed into the proposed artifact detection model to identify artifacts, followed by an optimization technique to select the best tiles—those with minimal artifacts and background and maximum qualified tissue. Finally, the WSI is reconstructed by combining the selected tiles to generate the final output. Additionally, the model generates a segmentation for the entire WSI and also provides statistics on the tile segmentations. </p>

- Place your Whole Slide Image (WSI) into the `test_wsi` folder
- The pre-trained weights for artifact detection are available in the `pretrained_ckpt` folder, while the weights for pen-marker removal are located in the `Ink_Removal/pre-trained` folder
- In the terminal execute:
  `python test_wsi.py`

- After running the inference, you will obtain the following outputs in `test_wsi` folder:
    - A thumbnail image of WSI
    - A thumbnail image of WSI with regions of interest (ROI) identified
    - A segmentation mask highlighting segmented regions of the WSI [Qualifed tissue: green, fold: red, blur: orange, and background: black]
    - A segmentation mask highlighting only qualified tissue regions of the WSI [background:0, qualified tissue:255]
    - Excel files contain statistics on identified artifacts
    - A folder named Selected_tiles containing qualified tiles
- If your WSI image has a format other than .svs or .mrxs, please modify line 92 in `test_wsi.py`
- It is recommended to use a tile size of 270 × 270 pixels
- To generate tiles of different sizes (e.g., 512x512):
    - Run the pipeline to generate the qualified tissue mask
    - Use the qualified tissue mask and the WSI to generate tiles of the desired size (a Python script will be provided soon to do this)
- If your WSI image contains pen-markings other than red, blue, green, or black, please update the `pens.py` file (located in the `wsi_tile_cleanup/filters folder`) to handle any additional pen-markings
- To generate a high-resolution thumbnail image and segmentation masks, you can adjust the `thumbnail_size` parameter in `inti_artifact.py`. However, note that this will increase the execution time
- To generate a folder containing all tile segmentation masks, comment out line 189 in the `test_wsi.py`
- Check out the useful parameters on line 58 of `inti_artifact.py` and adjust them if needed

# Training:

- To retrain the artifact detection model, refer to the details provided in: [GitHub](https://github.com/Falah-Jabar-Rahim/A-Fully-Automatic-DL-Pipeline-for-WSI-QA)
- To retrain the ink removal detection model, refer to the details provided in: [GitHub](https://github.com/Vishwesh4/Ink-WSI)

# Results & Benchmarking

![WSI-QA](./Figs/Fig.3.png)
![WSI-QA](./Figs/Fig.4.png)
![WSI-QA](./Figs/Fig.5.png)
![WSI-QA](./Figs/Fig.7.png)

<p align="justify"> Benchmark models, GrandQC (https://github.com/cpath-ukk/grandqc) pixel-wise segmentation model developed for artifact detection, and four tile-wise classification models with different network architectures (https://github.com/NeelKanwal/Equipping-Computational-Pathology-Systems-with-Artifact-Processing-Pipeline), namely MoE-CNN, MoE-ViT, multiclass-CNN, and multiclass-ViT. The proposed pixel-wise segmentation model is compared to GrandQC based on pixel segmentation accuracy and to MoE-CNN, MoE-ViT, Multiclass-CNN, and Multiclass-ViT based on tile classification. The classification considers three classes—artifact-free, fold, and blur. The model takes input tiles and generates segmentation masks, which are then used for tile classification. The classification process follows these criteria: (1) If the background occupies more than 50% of the tile, it is classified as a background tile. (2) If the background occupies less than 50%, but blurring and/or folding artifacts exceed 10% of the tile, it is classified as either fold or blur. (3) If the background is less than 50% and blurring and/or folding artifacts are below 10%, the tile is classified as artifact-free. The internal and external datasets are described in the manuscript. For segmentation, the ground truth segmentation masks are compared to the segmentation masks generated by the model. For classification, the predicted classes are compared to the ground truth labels. Quantitative metrics, including total accuracy (Acc), precision, recall, and F1 score, were used to evaluate classification performance, and the Dice metric was used to evaluate segmentation performance. The source code and model weights for benchmark models were obtained from the original GitHub repositories. The dataset includes tiles and their corresponding segmentation masks, each with a resolution of 270×270 pixels. The input tile size for the proposed segmentation model and MoE-CNN is the same and is resized to 288×288 to ensure compatibility with QrandQC </p>

# Notes:

- If your WSIs do not contain pen-marking artifacts, you can also use this pipeline: [GitHub](https://github.com/Falah-Jabar-Rahim/A-Fully-Automatic-DL-Pipeline-for-WSI-QA)
- WSI-SmartTiling is designed to clean and prepare WSIs for deep learning model development, prioritizing performance over efficiency
- The execution time for the proposed artifact detection, QrandQc, and MoE-CNN models on  NVIDIA GeForce RTX 4090 (24GB) running CUDA 12.2, Ubuntu 22.04,  32-core CPU, and 192GB of RAM are 6.39, 2.96, and 1.92 minutes, respectively, for a WSI at 20X magnification with dimensions (31,871 × 25,199) pixels
- The source code for the GUI interface described in the paper is located in the `Subjective-Evaluation-Interface` folder

# Acknowledgment:

Some parts of this pipeline were adapted from work on [GitHub](https://github.com/pengsl-lab/DHUnet) and [GitHub](https://github.com/Vishwesh4/Ink-WSI). If you use this pipeline, please make sure to cite their work properly

# Contact:

If you have any questions or comments, please feel free to contact: falah.rahim@unn.no


