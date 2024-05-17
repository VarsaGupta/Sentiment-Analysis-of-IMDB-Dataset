import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, roc_auc_score, f1_score
from datapreprocessing.datapreprocessing import DataCleaning, LemmaTokenizer
from dataloader.dataload import load_dataset
from evaluation.evaluationmetrics import confusion_matrix_plot

# Load dataset
data = load_dataset()

# Split data
x_train, x_test, y_train, y_test = train_test_split(data['Reviews'], data['Label'], test_size=0.01, random_state=42)

# Define the text classifier pipeline with vectorization
text_clf = Pipeline(steps=[
    ('clean', DataCleaning()),
    ('vect', TfidfVectorizer(analyzer="word", tokenizer=LemmaTokenizer(), ngram_range=(1, 3), min_df=10, max_features=10000)),
    ('clf', LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None))
])

# Train the model
text_clf.fit(x_train, y_train)

# Generate predictions on test data
y_predict = text_clf.predict(x_test)
y_score = text_clf.predict_proba(x_test)[:, 1]

# Evaluate the model
precision = precision_score(y_test, y_predict, average='micro')
auc_score = roc_auc_score(y_test, y_score, multi_class='ovo', average='macro')
f1 = f1_score(y_test, y_predict, average="weighted")

print("Precision Score on test dataset for Logistic Regression: %s" % precision)
print("AUC Score on test dataset for Logistic Regression: %s" % auc_score)
print("F1 Score on test dataset for Logistic Regression: %s" % f1)

# Plot confusion matrix
confusion_matrix_plot(y_test, y_predict)


# Save the trained model
model_dir = os.path.join(os.getcwd(), 'sentimentanalysis', 'models', 'model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'model_2.pkl')
joblib.dump(text_clf, model_path, compress=True)


print("Model trained and saved successfully.")
