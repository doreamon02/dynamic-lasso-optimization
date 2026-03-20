# Dynamic Soft-Thresholding for Feature Selection in High-Dimensional Regression
## Project Overview
This project explores feature selection in regression using a modified version of LASSO. Instead of using a fixed regularization parameter, the model applies dynamic soft-thresholding, allowing the penalty to change during training.
The method is evaluated on the House Prices dataset and compared with Ridge and standard LASSO regression to understand its impact on both prediction accuracy and feature selection.
## Objective
- Predict house prices using regression models  
- Reduce unnecessary features using L1 regularization  
- Improve standard LASSO by introducing a dynamic threshold  
- Compare results with baseline models  
## Dataset
- Dataset: House Prices (Kaggle)  
- Target variable: `SalePrice`  
- Contains both numerical and categorical features  
## Methodology
### Baseline Models
- Ridge Regression  
- Standard LASSO  
### Proposed Approach
The model modifies LASSO by updating the regularization parameter at each iteration:
λ(k) = λ₀ / (1 + γk)
This approach:
- Removes weak features early in training  
- Allows better refinement of important features later  
## Algorithm
1. Initialize coefficients to zero  
2. Compute gradient of the loss function  
3. Perform a gradient descent step  
4. Apply soft-thresholding  
5. Update the threshold dynamically  
6. Repeat until convergence  
## Evaluation Metrics
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  
- Number of selected features  
## Results
The results show that:
- Ridge produces stable predictions but retains most features  
- Standard LASSO performs feature selection but may remove useful correlated variables  
- Dynamic LASSO provides a balance between sparsity and accuracy  
## Visualizations
- RMSE comparison across models  
- Actual vs predicted values  
- Feature selection comparison  
- Coefficient distribution  
## Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## How to Run

Clone the repository:
```bash
git clone <your-repo-link>
cd <your-repo-folder>
