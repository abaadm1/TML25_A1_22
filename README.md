The development of this Membership Inference Attack (MIA) model followed an iterative and experimental approach. We began with simpler classifiers to understand how well the attack features captured membership signals.
Approach 1: Baseline with Logistic Regression
Started with a basic Logistic Regression model using features such as loss, confidence, and entropy. This served as a sanity check to verify that the extracted features carried useful signals.
Result: While the model showed some predictive power, the overall ROC AUC and TPR@FPR metrics were modest. It became clear that linear models couldn't capture the complexity in the data.

Approach 2: Exploring Tree-Based Models
In our next approach we moved on to using a Random Forest Classifier. Tree-based models are better at modelling non-linear relationships and interactions between features.
Result: Although this improved performance slightly, the model still struggled with generalizing across validation folds. The prediction probabilities were also poorly calibrated, which negatively affected the TPR@FPR metric used for evaluation.

Approach 3: Ensemble Method
To boost performance, we experimented with a voting ensemble combining Logistic Regression and XGBoost. This approach helped balance interpretability and power by blending a linear and a non-linear model.
Result: The ensemble gave better results than the individual models, but it was hard to calibrate and tune effectively. Additionally, the probabilistic outputs remained suboptimal for threshold-based evaluation metrics.
Final Approach: Calibrated XGBoost
1.	Model and Data Loading
•	Loads a pretrained ResNet18 model trained on a classification task with 44 classes.
•	Loads both public and private datasets serialized as MembershipDataset objects.
•	Applies custom normalization transform to all input images.
2. Feature Engineering
Each sample is passed through the model to compute a set of informative statistics that are sensitive to membership:
•	Loss: Cross-entropy loss for the true label.
•	Margin: Gap between top-1 and top-2 predicted probabilities.
•	True Logit: Logit of the correct class.
•	Gradient Norm: L2 norm of gradients w.r.t. the input image.
•	Confidence: Max softmax probability.
•	Entropy: Total uncertainty of the model's prediction.
•	Correctness: Whether the prediction was correct.
•	Prob. Std / Entropy / Top-k: Distribution properties of prediction.
These features are extracted for both public and private datasets.
3. Feature Selection
To reduce dimensionality and focus on the most informative signals:
•	We compute absolute Pearson correlation between each feature and the membership label.
•	The top 7 features (excluding the label itself) are retained for training.
This ensures that the model focuses on high-signal features without overfitting to noise.
4. Model Training with Calibration
•	A Stratified 5-Fold Cross-Validation scheme ensures balanced evaluation across folds.
•	For each fold:
o	An XGBoost classifier is trained.
o	It is wrapped in a CalibratedClassifierCV using isotonic regression for probability calibration.
o	Evaluation is done using ROC AUC and TPR@FPR=0.05, a strong privacy metric.
5. Final Model and Evaluation
•	Retrain the calibrated model on the entire public dataset using the best parameters and selected features.
•	Save the model, scaler, and feature list for later inference.
•	Evaluate on full public data to report AUC and TPR@FPR metrics.
6. Private Data Inference
•	Features are extracted from private data using the same method.
•	Features are scaled using the saved scaler.
•	Membership scores are predicted using the trained model.
•	Scores are clipped to [0.001, 0.999] to avoid extreme predictions.

To get an improvement in the result we also tried adding the data augmentation. For each sample, I added small noise to the image multiple times and measured how often the model’s predicted class stayed the same. The calculated correlation with the membership label and selected the top 7 features. Unfortunately, this did not show any improvement. The final score of AUC we achieved on the scoreboard is 0.644 and the TPR achieved on the scoreboard is 0.081. 


