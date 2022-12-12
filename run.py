import continual
import os

client = continual.Client(api_key="apikey/dc534806630148a3b5d567405c908d26")
run_id = os.environ.get("CONTINUAL_RUN_ID", None)
run = client.runs.create(description="An example run", run_id=run_id)
model = run.models.create("test_model")
model_version = model.model_versions.create()
dataset = run.datasets.create("test_dataset")
input_f = dataset.input_features()
print(input_f)
#run.datasets.list()

run.complete()
