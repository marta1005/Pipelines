import ml_deploy
import mlflow

class Mlflow_MLDeploy(mlflow.pyfunc.PythonModel, ml_deploy.MLModelProcess):

    def load_context(self, context):
        self.json_file = context.artifacts['json']
        self.model=ml_deploy.MLModelProcess(self.json_file, debug=False)
        self.model.read_model_process()
        self.model.build()
        
    def predict(self, context, input, *kwargs):
        return self.model.predict(input, uncertainty_management =None, *kwargs)