
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load results
predictions = pd.read_csv('your_submission.csv')
ground_truth = pd.read_csv('practice_solution.csv')

# Calculate metrics
accuracy = accuracy_score(ground_truth['true_label'], predictions['prediction'])
print(f"Accuracy: {accuracy:.3f}")
print(f"Expected for 85th percentile: >0.750")

# Detailed report
print(classification_report(ground_truth['true_label'], predictions['prediction']))
