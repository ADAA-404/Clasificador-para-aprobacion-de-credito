[Versión en Español](README.md)

# Bank Credit Prediction Using ML Techniques 🗳

This project implements a Machine Learning model to predict the probability of credit card customer payment default. The key problem this project aims to solve is helping financial institutions make more informed decisions about credit approvals, reducing the risk of losses due to default, and optimizing portfolio management. A Random Forest classifier is trained and optimized for this purpose, using historical customer data.

## Libraries Used 🐍
- pandas: Data manipulation, cleaning, and analysis for tabular data (structuring the customer dataset).
- numpy: Efficient numerical operations, especially with data arrays.
- matplotlib: Creation of static visualizations (bar charts, confusion matrices).
- seaborn: Creation of more attractive and complex statistical visualizations (count plots).
- scikit-learn (sklearn): Data preprocessing (handling imbalance with resample, One-Hot encoding with get_dummies) and ETL.
- xlrd: A dependency required by pandas to read older Excel files (.xls).
- warnings: Controls the display of warnings during script execution.

## Installation Considerations ⚙️
If you're using pip:

pip install -q 

  numpy==1.26.4 \
    
  pandas==2.2.2 \
    
  matplotlib==3.9.0 \
    
  seaborn==0.13.2 \
    
  scikit-learn==1.4.2 \
    
  xlrd==2.0.1

For this project, the code was written in Jupyter Notebook for Python.

## Usage Example 📎
The script performs a complete Machine Learning workflow, from data loading and cleaning to training and optimizing a classification model.
 1. Data Loading: The credit card customer dataset is loaded directly from a URL.
 2. Data Preprocessing: Columns are renamed for clarity; class balancing is performed using downsampling to equalize the number of customer instances.
 3. Data Splitting: The dataset is divided into training and testing sets (70% for training, 30% for testing).
 4. Base Model Training and Evaluation: A RandomForestClassifier with initial hyperparameters is used, and the confusion matrix is visualized to understand the model's performance on each class.
 5. Hyperparameter Optimization (Randomized Search): RandomizedSearchCV is used to find the best hyperparameters (max_depth, min_samples_split, min_samples_leaf) for the RandomForestClassifier.
    
## Contributions 🖨️
If you're interested in contributing to this project or using it independently, consider:
- Forking the repository.
- Creating a new branch (git checkout -b feature/new-feature).
- Making your changes and committing them (git commit -am 'Add new feature').
- Pushing your changes to the branch (git push origin feature/new-feature).
- Opening a 'Pull Request'.

## License 📜
This project is under the MIT License. Refer to the LICENSE file (if applicable) for more details.
