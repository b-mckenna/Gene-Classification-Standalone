from continual import Client
import os
import warnings
import argparse
import pandas as pd
#from continual.python.sdk.utils import get_dataset_stats_entry_dict
from continual.python.sdk.runs import Run
# from continual import DataCheck
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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

def calculate_metrics(y_test,y_pred):
	accuracy = accuracy_score(y_test,y_pred)
	precision=precision_score(y_test,y_pred,average='weighted')
	recall=recall_score(y_test,y_pred,average='weighted')
	f1=f1_score(y_test,y_pred,average='weighted')
	return setup_metrics_dict(accuracy, f1, recall, precision)

def create_confusion_matrix(name, y_test,y_pred):
	plt.figure(figsize=(10,8))
	sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='viridis')
	name = name
	plt.savefig(name + ".png")

def run_experiment(experiment, name, model, X_train, y_train, y_test):
	model.fit(X_train,y_train)

	joblib.dump(model, open(name+"_model", "wb"))

	# Test model
	y_pred=model.predict(X_test)

	metrics_dict = calculate_metrics(y_test, y_pred)
	
	create_confusion_matrix(name+"_confusion_matrix", y_test, y_pred)

	# Log confusion matrix, model, test metrics, and tags to experiment
	experiment.artifacts.create(key=name+"_confusion_matrix", path="./"+name+"_confusion_matrix.png", type="graph")
	experiment.artifacts.create(name+'_model',name+'_model', external=False, upload=True)
	for i in metrics_dict:
		experiment.metrics.create(key=i["key"], value=i["value"], direction=i["direction"], group_name=i["group_name"])

	experiment.tags.create(key="algo", value=name)

def get_metric_id(experiment, key):
    for exp in experiment.metrics.list(page_size=10):
        if exp.key == key:
            return exp.value

#exp.name.split('/')[9]
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
		parser.add_argument('--fit_prior', type=bool, default=False)

		args, _ = parser.parse_known_args()
	except Exception as e:
		logger.exception(
            "Unable to parse CLI arguments Error: %s", e
        )
	try:
		client = Client(api_key="apikey/346585479d59413b89a5996885e26563") #036dd4d42e94451d97c9bd7bfbb004fa
		client.set_config_project(project="projects/dna_sequencing", save=False, environment="production")
		run_id = os.environ.get("CONTINUAL_RUN_ID", None)
		run = client.runs.create(description="An example run", run_id=run_id)
		run.state == "ACTIVE"
	except Exception as e:
		logger.exception(
            "Unable to create a client and run. Error: %s", e
        )
	
	#run.start_run()
	id = "dna_sequencing_model"
	model = run.models.create(display_name=id, description="Classifying dna into gene families")
	model_version = model.model_versions.create()

	# Create dataset object and load data from local text file
	dna_dataset = run.datasets.create("DNA")
	dataset_version = dna_dataset.dataset_versions.create()
	dna_data = pd.read_table('dna_sequence_dataset/human.txt')

	# Profile data
	# TODO refactor with https://help.continual.ai/sdk-reference/#continual.python.sdk.data_profiles.DataProfilesManager.create

	#dataset_stats = get_dataset_stats_entry_dict(
	#	dna_data,
	#	"sequence", 
	#	["class"],
	#	"class",
	#	"class"
	#	)
	#dataset_version.create_data_profile(stats_entries=[dataset_stats])

	# Check data
	#checks = [dict(display_name = "my_data_check", success=True)]
	#dataset_version.create_data_checks(checks)
	#print(dataset_version.list_data_checks(page_size=3))

	# Data Transformation
	X, y = transform(dna_data)

	# Split dataset
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=10)
	xgb_params = {
			'max_depth': args.max_depth,
			'eta': args.learning_rate,
			'num_class': args.num_class,
			'eval_metric': args.eval_metric,
			'reg_alpha': args.alpha
	}
	xgb=XGBClassifier(xgb_params)
	
	# Create experiment
	xgb_experiment = model_version.experiments.create()
	
	run_experiment(xgb_experiment, "xgb", xgb, X_train, y_train, y_test)
	
	# Train second algorithm
	mnb = MultinomialNB(alpha=args.alpha,fit_prior=args.fit_prior)
	
	mnb_experiment = model_version.experiments.create()
	run_experiment(mnb_experiment, "mnb", mnb, X_train, y_train, y_test)
	
	# Compare experiments to find which is better performing
	xgb_accuracy = get_metric_id(xgb_experiment, "accuracy")
	mnb_accuracy = get_metric_id(mnb_experiment, "accuracy")
	print(xgb_accuracy)

	# TODO: Register winning model
	if mnb_accuracy > xgb_accuracy:
		# Log mnb model artifact to model_version
		model_version.artifacts.create('mnb_model','mnb_model', upload=True, external=False)
	elif mnb_accuracy < xgb_accuracy:
		# Log xgb model artifact to model_version
		model_version.artifacts.create('xgb_model','xgb_model', upload=True, external=False)
	else:
		# Log mnb model artifact to model_version
		model_version.artifacts.create('mnb_model','mnb_model', upload=True, external=False)

	run.complete()