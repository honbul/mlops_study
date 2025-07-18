import kfp
from kfp import dsl
from kfp import kubernetes

@dsl.container_component
def train_timm_imagenet_tiny(
    data_dir: str,
    output_dir: dsl.OutputPath(str),
):
    return dsl.ContainerSpec(
        image='localhost:5000/timm-jupyter:cuda12.1',
        command=["python", "-c"],
        args=[
            (
                "import os, sys, subprocess\n"
                "data_dir = sys.argv[1]\n"
                "output_path = sys.argv[2]\n"
                "print('CWD:', os.getcwd())\n"
                "print('List /mnt/data:', os.listdir('/mnt/data') if os.path.exists('/mnt/data') else '없음')\n"
                "print('List /mnt/data/tiny-imagenet-200:', os.listdir('/mnt/data/tiny-imagenet-200') if os.path.exists('/mnt/data/tiny-imagenet-200') else '없음')\n"
                "subprocess.run([\n"
                "    'python', '/workspace/timm/train.py',\n"
                "    data_dir,\n"
                "    '--data-dir', data_dir,\n"
                "    '--model', 'efficientnet_b0',\n"
                "    '--num-classes', '200',\n"
                "    '--input-size', '3', '64', '64',\n"
                "    '--epochs', '5',\n"
                "    '-b', '64',\n"
                "    '--crop-pct', '1.0',\n"
                "    '--log-interval', '50',\n"
                "    '--output', '/tmp/train_output',\n"
                "    '-j', '8',\n"
                "    '--pin-mem'\n"
                "], check=True)\n"
                "with open(output_path, 'w') as f:\n"
                "    f.write('/tmp/train_output')\n"
            ),
            data_dir,
            output_dir
        ]
    )

@dsl.pipeline(
    name="timm_pipeline_v2_pvc",
    description="Train timm with TinyImageNet using PVC in KFP v2"
)
def timm_pipeline_v2_pvc(
    pvc_data_dir: str = "/mnt/data",
    pvc_name: str = "tinyimagenet-pvc",
):
    # Step 1: Training
    train = train_timm_imagenet_tiny(
        data_dir=pvc_data_dir,
    )

    # Step 2: PVC Mount (kfp-kubernetes에서 제공)
    kubernetes.mount_pvc(
        train,
        pvc_name=pvc_name,
        mount_path="/mnt/data"
    )

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=timm_pipeline_v2_pvc,
        package_path='timm_pipeline_v2_pvc.yaml'
    )
