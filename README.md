# MLOps Study: Kubeflow, MLflow, and PyTorch (timm)

This repository contains resources and configurations for an MLOps study project. It integrates **Kubeflow Pipelines**, **MLflow**, and **PyTorch Image Models (timm)** to create a scalable model training and tracking environment on Kubernetes.

## 📂 Directory Structure

- **`code/`**: Contains the model training code.
  - `pytorch-image-models/`: A clone/version of the [timm](https://github.com/huggingface/pytorch-image-models) library used for training `efficientnet_b0` and other models.
- **`docker/`**: Dockerfiles and Compose files.
  - `create_base_image.Dockerfile`: Base image setup.
  - `create_kubeflow_jupyter.Dockerfile`: Image for Kubeflow Jupyter notebooks.
  - `docker-compose_mlflow.yaml`: Docker Compose setup for a local MLflow server.
- **`k8s/`**: Kubernetes manifests and pipeline definitions.
  - `base/`: Persistent Volume (PV) and PVC definitions for datasets (TinyImageNet) and artifacts.
  - `jobs/`: Standalone Kubernetes Jobs for debugging or single runs.
  - `pipelines/`: Python scripts using the Kubeflow Pipelines (KFP) SDK to define training workflows.
  - `monitoring/`: Manifests for monitoring (e.g., DCGM, NVIDIA PodMonitor).
- **`charts/`**: Helm chart value overrides (e.g., for MLflow).
- **`kubeflow/`**: Kubeflow specific manifests.

## 🚀 Getting Started

### Prerequisites

- A running **Kubernetes Cluster** (with GPU support if training on GPU).
- **Kubeflow** installed on the cluster.
- **Docker** for building images.
- **Python 3.x** and `kfp` SDK installed locally.

### 1. Infrastructure Setup

Ensure your Kubernetes cluster has the necessary storage classes and permissions. Apply the base manifests to create Persistent Volume Claims (PVCs) for datasets and MLflow artifacts.

```bash
kubectl apply -f k8s/base/
```

*Note: You may need to adjust storage paths in the PV definitions to match your environment.*

### 2. Build Docker Images

Build the custom Docker images required for the training jobs and notebooks.

```bash
# Example for building the Kubeflow Jupyter image
docker build -t localhost:5000/timm-jupyter:cuda12.1 -f docker/create_kubeflow_jupyter.Dockerfile .
docker push localhost:5000/timm-jupyter:cuda12.1
```

### 3. Setup MLflow

You can run MLflow using Docker Compose for local testing or deploy it to Kubernetes.

**Local (Docker Compose):**
```bash
cd docker
docker-compose -f docker-compose_mlflow.yaml up -d
```

**Kubernetes:**
Use the provided manifests or Helm charts in `charts/` to deploy MLflow to your cluster.

### 4. Run Kubeflow Pipelines

The pipelines are defined in `k8s/pipelines/`. Use the Python scripts to compile them into YAML format.

**Example: TinyImageNet Training Pipeline**

1.  Compile the pipeline:
    ```bash
    python k8s/pipelines/tinyimagenet_timm_base_pipeline.py
    ```
    This will generate `timm_pipeline_v2_pvc.yaml`.

2.  Upload the generated YAML file to the Kubeflow Pipelines UI.
3.  Create a run, specifying necessary parameters (e.g., PVC names).

## 🛠️ Workflows

### Training with `timm`

The pipeline uses the `timm` library to train models (e.g., `efficientnet_b0`) on the **Tiny ImageNet** dataset. The training script is executed within a container that mounts the dataset via PVC.

Key Pipeline Steps:
1.  **Data Mounting**: Mounts the pre-loaded PVC containing the dataset to `/mnt/data`.
2.  **Training**: Executes `train.py` from the `timm` library with specified hyperparameters.
3.  **Output**: Artifacts and logs are stored or tracked (via MLflow integration if enabled).

## 📝 License

This project includes code from the `pytorch-image-models` (timm) library, which is Apache 2.0 licensed.
