#!/bin/bash

NFS_ROOT="/srv/nfs/kube-share/kubeflow"

# 777이 아닌 모든 하위 디렉토리에 대해 777 적용
find "$NFS_ROOT" -type d ! -perm 0777 -exec chmod 777 {} +
