from continual import Client
import os
import warnings
import argparse
import pandas as pd
from continual.python.sdk.utils import get_dataset_stats_entry_dict
from continual import DataCheck
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


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

def setup_metrics_dict(accuracy, f1, recall, precision):
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
	return metric_dicts

if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	try:
		parser = argparse.ArgumentParser()
		# Set hyperparams via CLI arguments
		parser.add_argument('--alpha', type=float, default=0.5)
		parser.add_argument('--learning_rate', type=float, default=0.1)
		parser.add_argument('--max_depth', type=int, default=4.0)
		parser.add_argument('--eval_metric', type=str, default="mae")
		parser.add_argument('--num_class', type=int, default=3)

		args, _ = parser.parse_known_args()
	except Exception as e:
		logger.exception(
            "Unable to parse CLI arguments Error: %s", e
        )
	try:
		client = Client(api_key="apikey/dc534806630148a3b5d567405c908d26")
		run_id = os.environ.get("CONTINUAL_RUN_ID", None)
		run = client.runs.create(description="An example run", run_id=run_id)
	except Exception as e:
		logger.exception(
            "Unable to create a client and run. Error: %s", e
        )
	
	model = run.models.create("test_model")
	model_version = model.model_versions.create()

	# Create dataset object and load data from local text file
	dna_dataset = run.datasets.create("DNA")
	dataset_version = dna_dataset.dataset_versions.create()
	dna_data = pd.read_table('dna_sequence_dataset/human.txt')

	# Profile data
	dataset_stats = get_dataset_stats_entry_dict(
		dna_data,
		"sequence", 
		["class"],
		"class",
		"class"
		)
	dataset_version.create_data_profile(stats_entries=[dataset_stats])

	# Check data
	checks = [dict(display_name = "my_data_check", success=True)]
	dataset_version.create_data_checks(checks)
	print(dataset_version.list_data_checks(page_size=3))

	# Data Transformation
	X, y = transform(dna_data)

	# Split dataset
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=10)

	# Set parameters
	params ={
		'max_depth': args.max_depth,
		'eta': args.learning_rate,
		'num_class': args.num_class,
		'eval_metric': args.eval_metric,
		'reg_alpha': args.alpha
	}
	epochs = 5

	# Train model
	xgb=XGBClassifier(params)
	xgb.fit(X_train,y_train)

	# Test model
	y_pred=xgb.predict(X_test)

	accuracy = accuracy_score(y_test,y_pred)
	f1=f1_score(y_test,y_pred,average='weighted')
	recall=recall_score(y_test,y_pred,average='weighted')
	precision=precision_score(y_test,y_pred,average='weighted')

	metric_dicts = setup_metrics_dict(accuracy, f1, recall, precision)
	
	# Log metrics from previous model test
	model_version.create_metrics(metrics=metric_dicts)

	print(model_version.list_metrics(page_size=3))

	# Create confusion matrix
	plt.figure(figsize=(10,8))
	sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='viridis')
	plt.savefig('confusion_matrix.png')

	# Log confusion matrix
	confusion_matrix_artifact = model_version.artifacts.create('confusion_matrix','confusion_matrix.png', upload=True, external=False)

	# Create experiment
	experiment = model_version.experiments.create()

	# Log confusion matrix, model, test metrics, and tags to experiment
	experiment.artifacts.create(key="confusion_matrix", path="./confusion_matrix.png", type="graph")
	experiment.artifacts.create('xgb_model','xgb_model')
	experiment.create_metrics(metrics=metric_dicts)
	experiment.tags.create(key="name", value="higher_alpha")

	# Print out experiment
	print([exp for exp in model_version.experiments.list_all()])

	# Save XG Boost model as pickle file
	pickle.dump(xgb, open("xgb_model", "wb"))

	# Log xgb model artifact to model_version
	model_version.artifacts.create('xgb_model','xgb_model', upload=True, external=False)

	# Load local xgb model artifact
	xgb_model_loaded = pickle.load(open('xgb_model', "rb"))

	run.complete()

