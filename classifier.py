import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import joblib

# Load extracted features
df = pd.read_csv("public_features.csv")

# Features and labels
X = df[["confidence", "loss", "entropy"]].values
y = df["membership"].values

# Train-test split for evaluation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[INFO] Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Train logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_val)
y_score = clf.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_score)

print(f"[RESULT] Validation Accuracy: {acc:.4f}")
print(f"[RESULT] Validation AUC: {auc:.4f}")

# Optionally view ROC curve threshold @ FPR = 0.05
fpr, tpr, thresholds = roc_curve(y_val, y_score)
idx = np.searchsorted(fpr, 0.05, side="right")
print(f"[DEBUG] TPR@FPR=0.05: {tpr[idx-1]:.4f} (if exists)")

# Save model
joblib.dump(clf, "attack_model.pkl")
print("[INFO] Saved trained attack model to attack_model.pkl")


####################################################################################################

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
# import joblib

# # Load extracted features
# df = pd.read_csv("public_features.csv")

# # Features and labels
# X = df[["confidence", "loss", "entropy"]].values
# y = df["membership"].values

# # Train-test split
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# print(f"[INFO] Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# # Train Random Forest
# clf = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=5,
#     random_state=42
# )
# clf.fit(X_train, y_train)

# # Predict
# y_pred = clf.predict(X_val)
# y_score = clf.predict_proba(X_val)[:, 1]

# # Evaluation
# acc = accuracy_score(y_val, y_pred)
# auc = roc_auc_score(y_val, y_score)

# print(f"[RESULT] Validation Accuracy: {acc:.4f}")
# print(f"[RESULT] Validation AUC: {auc:.4f}")

# fpr, tpr, thresholds = roc_curve(y_val, y_score)
# idx = np.searchsorted(fpr, 0.05, side="right")
# if idx > 0:
#     print(f"[DEBUG] TPR@FPR=0.05: {tpr[idx - 1]:.4f}")
# else:
#     print("[DEBUG] TPR@FPR=0.05: Not found in range")

# # Save model
# # joblib.dump(clf, "attack_model.pkl")
# # print("[INFO] Saved trained attack model to attack_model.pkl")


##########################################################################################

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
# import xgboost as xgb
# import joblib

# # Load extracted features
# df = pd.read_csv("public_features.csv")

# # Features and labels
# X = df[["confidence", "loss", "entropy"]].values
# y = df["membership"].values

# # Train-test split
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# print(f"[INFO] Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# # XGBoost classifier
# clf = xgb.XGBClassifier(
#     n_estimators=300,
#     max_depth=4,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     use_label_encoder=False,
#     eval_metric='logloss',
#     random_state=42
# )
# clf.fit(X_train, y_train)

# # Predict
# y_pred = clf.predict(X_val)
# y_score = clf.predict_proba(X_val)[:, 1]

# # Evaluation
# acc = accuracy_score(y_val, y_pred)
# auc = roc_auc_score(y_val, y_score)

# print(f"[RESULT] Validation Accuracy: {acc:.4f}")
# print(f"[RESULT] Validation AUC: {auc:.4f}")

# # TPR@FPR=0.05
# fpr, tpr, thresholds = roc_curve(y_val, y_score)
# idx = np.searchsorted(fpr, 0.05, side="right")
# if idx > 0:
#     print(f"[DEBUG] TPR@FPR=0.05: {tpr[idx - 1]:.4f}")
# else:
#     print("[DEBUG] TPR@FPR=0.05: Not found in range")

# # Save model
# # joblib.dump(clf, "attack_model.pkl")
# # print("[INFO] Saved trained attack model to attack_model.pkl")
