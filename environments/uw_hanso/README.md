# UW HANSO legacy runtime

Build this environment only on Linux amd64 after the trained model artifacts and terms of use have
been obtained. Do not attempt to install this dependency stack into the primary `ards-nlp`
environment or the native macOS arm64 runtime.

This is the pinned inference subset of the upstream environment. The upstream repository also
lists `allennlp-models==1.3.0`, but the inference graph does not import it and that package requires
PyTorch 1.7 or newer, conflicting with the published `torch==1.6.0` runtime used here.

```bash
docker build --platform linux/amd64 \
  -t ards-nlp-uw-hanso:legacy \
  environments/uw_hanso
```

The container must be run locally with read-only mounts for the pinned upstream repository and
restricted comparator input packet. It must not send report text to a remote endpoint.
