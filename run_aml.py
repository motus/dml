"""
Runnable script to submit a test DML job to AzureML.
"""

import azureml.core
import azureml.train.estimator
import azureml.train.dnn


def _main():

    workspace = azureml.core.Workspace.from_config()

    estimator = azureml.train.estimator.Estimator(
        compute_target=workspace.compute_targets["nsnet-nv24"],
        user_managed=False,
        use_gpu=True,
        node_count=4,
        distributed_training=azureml.train.dnn.Mpi(process_count_per_node=1),
        source_directory=".",
        entry_script="run.py",
        script_params={})

    exp = azureml.core.Experiment(workspace=workspace, name="DML-MPI")
    run = exp.submit(estimator)

    print("DML-MPI experiment submitted: %s" % run.get_details())


if __name__ == "__main__":
    _main()
