# hash:sha256:ff3af49bd996ff3a7f1bb358f3a043babbeafa7ebe3a4ddf45431c3deaeb0b68
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG API_KEY
ARG API_SECRET

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    cebra[dev,demos]==0.4.0 \
    matplotlib==3.8.3 \
    numpy==1.26.4 \
    pandas==2.2.1 \
    scikit-learn==1.4.1.post1 \
    seaborn==0.13.2
