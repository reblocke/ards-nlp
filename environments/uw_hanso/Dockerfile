FROM continuumio/miniconda3:24.11.1-0

WORKDIR /workspace
RUN apt-get update \
    && apt-get install --yes --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean --all --yes
RUN conda run -n uw-hanso-ards python -c \
    "import matplotlib, medspacy, scipy, seaborn, sklearn, tensorboardX; import spacy; spacy.load('en_core_web_sm')"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "uw-hanso-ards"]
