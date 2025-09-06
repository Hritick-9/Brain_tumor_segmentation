# Brain Tumor Segmentation Using U-Net

This project demonstrates brain tumor segmentation from MRI scans using a U-Net deep learning architecture. The workflow includes data preparation, visualization, model training, and evaluation.

## Dataset
- **Source:** [BraTS 2020 Dataset](https://www.kaggle.com/awsaf49/brats20-dataset-training-validation)
- Downloaded using Kaggle API and unzipped in the workspace.

## Workflow
1. **Setup**
    - Install required packages (Kaggle, TensorFlow, Keras, etc.)
    - Configure Kaggle API for dataset download.
2. **Data Preparation**
    - Load and preprocess MRI scans (FLAIR, T1, T1ce, T2) and segmentation masks.
    - Normalize images and visualize slices.
    - Split data into training, validation, and test sets.
3. **Data Generator**
    - Custom Keras `DataGenerator` for efficient batch loading and augmentation.
4. **Model Architecture**
    - U-Net model built using TensorFlow/Keras for multi-class segmentation.
    - Loss functions and metrics: Dice coefficient, Mean IoU, Precision, Sensitivity, Specificity.
5. **Training**
    - Model trained for 35 epochs with callbacks for checkpointing and learning rate reduction.
    - Training logs saved for visualization.
6. **Evaluation & Visualization**
    - Model evaluated on test set.
    - Segmentation results visualized for random test samples and slices.
    - Metrics plotted: Accuracy, Loss, Dice coefficient, Mean IoU.

## Usage
1. Clone the repository and open the notebook `brain-tumor-segmentation-using-u-net (1).ipynb`.
2. Ensure you have a valid `kaggle.json` API key in your home directory or as specified in the notebook.
3. Run the notebook cells sequentially to:
    - Download and prepare the dataset
    - Train the U-Net model
    - Visualize results and evaluate performance

## Requirements
- Python 3.7+
- TensorFlow, Keras, scikit-image, OpenCV, Nibabel, Matplotlib, Pandas, Scikit-learn
- Kaggle API key

## References
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [BraTS Challenge](https://www.med.upenn.edu/cbica/brats2020/)

## License
This project is for educational purposes. Please check dataset and code licenses before commercial use.
