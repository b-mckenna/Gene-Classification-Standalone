from continual import Client
import os
import pandas as pd
from continual.python.sdk.utils import get_dataset_stats_entry_dict
from continual import DataCheck
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

def getKmers(sequence, size=7):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def transform(human_data):
	human_data['words']=human_data['sequence'].apply(lambda x: getKmers(x))
	human_data_revised=human_data.drop(columns='sequence',axis=1)
	human_texts = list(human_data_revised['words'])
	for item in range(len(human_texts)):
		human_texts[item] = ' '.join(human_texts[item])
		
	y=human_data_revised['class'].values
	cv = CountVectorizer(ngram_range=(4,4))
	X = cv.fit_transform(human_texts)
	return X, y


client = Client(api_key="apikey/dc534806630148a3b5d567405c908d26")

run_id = os.environ.get("CONTINUAL_RUN_ID", None)
run = client.runs.create(description="An example run", run_id=run_id)
model = run.models.create("test_model")
model_version = model.model_versions.create()

# DNA dataset
dna_dataset = run.datasets.create("DNA")
dataset_version = dna_dataset.dataset_versions.create()
dna_data = pd.read_table('dna_sequence_dataset/human.txt')

# Using Continual's Data Profiler
dataset_stats = get_dataset_stats_entry_dict(
	dna_data,
	"sequence", 
	["class"],
	"class",
	"class"
	)
dataset_version.create_data_profile(stats_entries=[dataset_stats])

# Data checks
checks = [dict(display_name = "my_data_check", success=True)]
dataset_version.create_data_checks(checks)
print(dataset_version.list_data_checks(page_size=3))

# Data Transformation
X, y = transform(dna_data)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=10)

# Train model
xgb=XGBClassifier()
xgb.fit(X_train,y_train)

# Test model
y_pred=xgb.predict(X_test)

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix

accuracy = accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred,average='weighted')
recall=recall_score(y_test,y_pred,average='weighted')
precision=precision_score(y_test,y_pred,average='weighted')

# Create metrics
metric_dicts = [
	dict(
		key="accuracy",
        value=accuracy,
        direction="HIGHER",
        group_name="test",
        step=1,
    ),
	dict(
		key="precision",
		value=precision,
		direction="HIGHER",
		group_name="test",
		step=1,
	),
	dict(
		key="f1",
		value=f1,
		direction="HIGHER",
		group_name="test",
		step=1,
    ),
	dict(
		key="recall",
		value=recall,
		direction="HIGHER",
		group_name="test",
		step=1,
	),
]

model_version.create_metrics(metrics=metric_dicts)

print(model_version.list_metrics(page_size=3))

# Data transformation
plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='viridis')
plt.savefig('confusion_matrix.png')

artifact = model_version.artifacts.create('confusion_matrix','confusion_matrix.png', upload=True, external=False)
#artifact.download()

run.complete()

