# Use the conda-forge base image with Python
FROM mambaorg/micromamba:jammy


# set environment variables
ENV PYTHONUNBUFFERED 1

RUN micromamba config append channels conda-forge
RUN micromamba config append channels openeye

COPY --chown=$MAMBA_USER:$MAMBA_USER  devtools/conda-envs/falcbot.yaml /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER  .  /home/mambauser/

RUN micromamba install -y -n base git -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /home/mambauser/FALCBot

RUN mkdir /openeye
ENV OE_LICENSE=/openeye/oe_license.txt


ENTRYPOINT [ "python", "falcbot/falcbot.py" ]