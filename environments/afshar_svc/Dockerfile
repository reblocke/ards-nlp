FROM continuumio/miniconda3:4.7.12

WORKDIR /workspace
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean --all --yes

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "afshar-ards-svc"]
