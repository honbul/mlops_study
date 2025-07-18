FROM localhost:5000/timm-mlflow:cuda12.1

ARG NB_USER=jovyan
ARG NB_UID=1000

RUN useradd -m -s /bin/bash -N -u ${NB_UID} ${NB_USER}

RUN pip install --no-cache-dir jupyterlab==4.2.1 jupyter_server kfp==2.11.0 jupyterlab-git==0.50.2

RUN mkdir -p /home/${NB_USER}/.jupyter && \
    echo "c.ServerApp.ip = '0.0.0.0'\n\
c.ServerApp.port = 8888\n\
c.ServerApp.open_browser = False\n\
c.ServerApp.allow_origin = '*'\n\
c.ServerApp.base_url = '${NB_PREFIX}'" \
    > /home/${NB_USER}/.jupyter/jupyter_server_config.py

COPY ./wrapper_mlflow.py /workspace/wrapper_mlflow.py

ENV HOME=/home/${NB_USER}
WORKDIR ${HOME}
USER ${NB_USER}
EXPOSE 8888
CMD ["jupyter", "lab"]
