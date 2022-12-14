import continual
import os
import pandas as pd 
import pandas_profiling
import json
from continual.python.sdk.utils import get_dataset_stats_entry_dict


client = continual.Client(api_key="apikey/dc534806630148a3b5d567405c908d26")
run_id = os.environ.get("CONTINUAL_RUN_ID", None)
run = client.runs.create(description="An example run", run_id=run_id)
model = run.models.create("test_model")
model_version = model.model_versions.create()

# DNA dataset
#dna_dataset = run.datasets.create("DNA")
#dataset_version = dna_dataset.dataset_versions.create()
#dna_data = pd.read_table('dna_sequence_dataset/human.txt')

# Energy Dataset
energy_dataset = run.datasets.create("AEP_hourly")
dataset_version = energy_dataset.dataset_versions.create()

human_data = pd.read_csv('AEP_hourly.csv')

# Using Continual's Data Profiler
dataset_stats = get_dataset_stats_entry_dict(
	human_data,
	"energy_data_profile", 
	["Datetime"],
	"Datetime",
	"Datetime"
	)
dataset_version.create_data_profile(stats_entries=[dataset_stats])

#dataset_version.data_profiles.get("dataset_stats")
#print("Test list")
#print(dataset_version.data_profiles.list())
print("Test list all")
print(dataset_version.data_profiles.list_all("dataset_stats"))

# Using the Pandas Profiler
#energy_dataset_report = pandas_profiling.ProfileReport(input)

#dataset_profiling_json = json.loads(energy_dataset_report.to_json())
#del dataset_profiling_json["table"]["types"]
#energy_profile = {
#	"name":dataset_profiling_json['analysis']['title'],
#	"datasetStats":list(dataset_profiling_json['table']),
#	"createTime":dataset_profiling_json["analysis"]["date_start"]
#}

#dataset_version.create_data_profile_from_dict(energy_profile)
#data_profiler_artifact = model_version.artifacts.create(energy_dataset_report)

# Data checks
data_checks = [{"column": "Datetime", "check": "range", "params": {"min": "2004-10-01", "max": '2018-08-03'}},
                   {"column": "AEP_MW", "check": "range", "params": {"min": 9581.0, "max": 25695.0}}]
dataset_version.create_data_check(data_checks)
print(dataset_version.list_data_checks(page_size=3))

# Data transformation
artifact = model_version.artifacts.create('my_artifact','fake_artifact.png', upload=True, external=False)
artifact.download("./artifacts")

run.complete()
