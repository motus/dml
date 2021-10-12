"""
Runnable script to submit NSNet inference job to AzureML.
"""

from azureml.core.conda_dependencies import CondaDependencies
from azureml.data.constants import WORKSPACE_BLOB_DATASTORE
import configargparse
import azureml.core
import azureml.train.estimator
import azureml.train.dnn
from azureml.core import Workspace, Dataset, ScriptRunConfig, Environment, Experiment
from azureml.core.runconfig import DockerConfiguration, PyTorchConfiguration, MpiConfiguration
from azureml.data.datapath import DataPath


def _main():

    workspace = Workspace.from_config(path=r"C:\Users\vigopal\source\repos\tools_av_models\audio\noise_suppression\config.json")
    curated_env_name = 'AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu'
    pytorch_env = Environment.get(workspace=workspace, name=curated_env_name)
    pytorch_env = pytorch_env.clone(new_name="pytorch-1.9-gpu-dml")
    pytorch_env.docker.enabled = True
    pytorch_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04'
    pytorch_env.docker.base_dockerfile = None
    pytorch_env.python.conda_dependencies = CondaDependencies.create(pip_packages=['ConfigArgParse',
                                                                             'numpy',
                                                                             'librosa',
                                                                             'pyYAML',
                                                                             'azureml-dataprep',
                                                                             'azureml-sdk',
                                                                             'onnxruntime',
                                                                             'pandas',
                                                                             'azureml-mlflow',
                                                                             'torchvision',
                                                                             'mpi4py'],
                                                               conda_packages=['pysoundfile', 'numpy'])
    distr_config = MpiConfiguration(process_count_per_node=1, node_count=4)


    run_config = ScriptRunConfig(
        source_directory= ".",
        script='run.py',
        compute_target=workspace.compute_targets["EvalNode"],
        environment=pytorch_env,
        distributed_job_config=distr_config,
        # arguments=script_arg_lst,
        docker_runtime_config=DockerConfiguration(use_docker=True)
        )

    exp = Experiment(workspace=workspace, name="dml")
    run = exp.submit(run_config)



if __name__ == "__main__":
    _main()
