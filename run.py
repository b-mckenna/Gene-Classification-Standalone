import continual
import os

client = continual.Client()
run_id = os.environ.get("CONTINUAL_RUN_ID", None)
run = client.runs.create(description="An example run", run_id=run_id)
model = run.models.create("test_model")
model_version = model.model_versions.create()
model_version.log_metrics(metrics=[
	dict(
			key="accuracy",
			value=0.8,
			group_name="test",
		)
])
model_version.promote(reason="UPLIFT")
run.complete()	
