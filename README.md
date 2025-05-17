Title:
Deep Learning and Red Deer Optimiser for Automatic Cardiovascular Disease Identification on MRI



DESCRIPTION
The proposed research presents an automated deep learning-based framework for the identification of cardiovascular diseases (CVD) using cardiac Magnetic Resonance Imaging (MRI). 
The system integrates multiple advanced techniques to enhance diagnostic precision and efficiency. Initially, image quality is improved using Wiener Filtering (WF) for noise reduction and Dynamic Histogram Equalization (DHE) for contrast enhancement. 
Segmentation of cardiac structures is performed using a U-Net model, followed by radiomics-based feature extraction to capture detailed quantitative patterns. 
The extracted features are classified using an Attention-based Convolutional Gated Recurrent Unit (ACGRU) network, which captures both spatial and temporal dependencies in the image data. To fine-tune model performance, the Red Deer Optimizer (RDO) is applied for optimal hyperparameter selection. 
The system is evaluated on a benchmark MRI dataset and demonstrates superior performance, achieving 95.66% accuracy and the lowest execution time of 1.254 seconds, outperforming traditional and deep learning methods. This research highlights a robust and efficient solution for automatic CVD detection, combining deep learning with metaheuristic optimization.




DATASET
The cardiac MRI dataset used in this study is sourced from the Kaggle repository titled “CAD Cardiac MRI Dataset.” It comprises labeled MRI image data organized into two distinct classes: Normal and Ill (patients with cardiovascular disease, CVD). The dataset contains a total of 1,820 cardiac MRI images, with 910 images representing normal cases and 910 images corresponding to ill cases, ensuring a balanced distribution between the two categories. 
This balanced structure supports unbiased training and evaluation of classification models for cardiovascular disease detection. The images are pre-processed grayscale scans suitable for deep learning applications in medical diagnosis.




CODE INFORMATION
The implementation of the proposed automated cardiovascular disease (CVD) detection framework is developed in Python 3.10.5 and organized into modular components to ensure clarity, reproducibility, and scalability. The code is structured as follows:
preprocessing.py
Implements Wiener Filtering (WF) and Dynamic Histogram Equalization (DHE) for MRI image enhancement.
Uses libraries: cv2, scikit-image, scipy.signal.

segmentation_unet.py
Contains the U-Net architecture used for segmenting cardiac regions from MRI images.
Built using TensorFlow and Keras.

feature_extraction.py
Performs radiomics feature extraction using statistical and texture descriptors (e.g., GLCM, intensity histograms).
Libraries: pyradiomics, numpy, pandas.

acgru_model.py
Implements the Attention-based Convolutional Gated Recurrent Unit (ACGRU) model for classifying normal vs. diseased images.
Combines CNN layers, Bi-GRU, and attention mechanisms.

red_deer_optimizer.py
Provides a custom implementation of the Red Deer Optimizer (RDO) for hyperparameter tuning of the ACGRU model.
Handles population initialization, master/stag/hind hierarchy, coupling logic, and fitness evaluation.

main.py
The main pipeline to execute the entire workflow: preprocessing → segmentation → feature extraction → training → evaluation.
utils.py 
Contains utility functions for loading data, calculating performance metrics (accuracy, precision, recall, F1-score, MCC), and visualizations (confusion matrix, ROC curves, etc.).


USAGE INSTRUCTIONS
Step 1: Environment Setup
Install Python 3.10.5 (if not already installed).
Create a virtual environment :
bash
Copy
Edit
python -m venv acvd_env
source acvd_env/bin/activate  # For Linux/Mac
acvd_env\Scripts\activate     # For Windows
Install required dependencies:
bash
Copy
Edit
pip install -r requirements.txt


Step 2: Dataset Preparation
Download the MRI dataset from Kaggle – Cardiac MRI Dataset or the link specified in the manuscript.
Unzip the dataset and organize it as follows:
bash
Copy
Edit
/data/
  ├── train/
  │   ├── normal/
  │   └── abnormal/
  ├── test/
  │   ├── normal/
  │   └── abnormal/
Ensure images are in .jpg or .png format for preprocessing and model compatibility.


Step 3: Running the Code
Launch the main pipeline:
bash
Copy
Edit
python main.py
This script will:
Preprocess images using WF and DHE.
Segment cardiac regions using U-Net.
Extract radiomics features.
Train and evaluate the ACGRU model with RDO optimization.

Expected Outputs:
Classification metrics: Accuracy, Precision, Recall, F1-Score, MCC.
Visualizations: Confusion matrix, ROC curve, PR curve.
Execution time and training logs.




REQUIREMENTS
To run the proposed cardiovascular disease detection system, the following dependencies and Python libraries are required. These should be listed in your requirements.txt file for easy installation.
1. Python Version
Python 3.10.5 or later

2. Core Libraries
txt
Copy
Edit
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
opencv-python>=4.5.3
scipy>=1.7.0

3. Deep Learning Libraries
txt
Copy
Edit
tensorflow>=2.8.0
keras>=2.8.0

4. Image Processing and Radiomics
txt
Copy
Edit
scikit-image>=0.18.3
pyradiomics>=3.0.1

5. Visualization 
txt
Copy
Edit
seaborn>=0.11.1
plotly>=5.3.1




METHODOLOGY
The proposed methodology presents a robust multi-stage deep learning pipeline named ACVD-RDODL for automatic cardiovascular disease (CVD) detection from cardiac MRI images. This framework integrates preprocessing, segmentation, feature extraction, classification, and metaheuristic optimization to ensure high diagnostic accuracy. The key stages are:
1. Image Preprocessing
Wiener Filtering (WF): Reduces noise and enhances signal-to-noise ratio.
Dynamic Histogram Equalization (DHE): Improves image contrast to highlight cardiac structures.
Goal: Enhance MRI quality before downstream processing.

2. Image Segmentation
Model Used: U-Net
Function: Automatically segments heart regions from MRI scans, isolating relevant anatomical areas for focused analysis.
Optimization: The segmentation is enhanced by applying the Red Deer Optimizer (RDO) to fine-tune U-Net parameters.

3. Feature Extraction
Technique: Radiomics
Features Extracted: Shape, intensity, texture (e.g., GLCM), and high-order statistical features.
Purpose: Quantify cardiac structure variations for better disease representation.

4. Image Classification
Model: Attention-based Convolutional Gated Recurrent Unit (ACGRU)
Convolutional Layers: Extract spatial features.
Bi-GRU Layers: Capture temporal dependencies and spatial variations.
Attention Mechanism: Highlights critical features dynamically across the sequence.
Output: Classifies MRI into Normal or Diseased categories.

5. Hyperparameter Optimization
Technique: Red Deer Optimizer (RDO)
Used For: Fine-tuning hyperparameters of the ACGRU model.
Advantage: Enhances convergence speed, avoids local minima, and boosts final classification accuracy.

6. Evaluation Strategy
Dataset: Kaggle Cardiac MRI dataset with Normal and CVD samples.
Metrics Used:
Accuracy, Precision, Recall (Sensitivity), Specificity
F1-Score, Matthews Correlation Coefficient (MCC)
Execution Time
Validation Method: Training/testing split with ROC, PR analysis, and confusion matrix.




CITATION
[26]	https://www.kaggle.com/code/kerneler/starter-cad-cardiac-mri-dataset-966083ec-e/input


# Supplementary-files
