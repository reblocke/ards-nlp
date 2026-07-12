# UW HANSO model acquisition

The public UW BioNLP repository does not include the trained model. Restricted-data inference is
blocked until both model files and their terms of use are documented.

| Field | Value |
|---|---|
| Request date | TODO |
| Contact | Professor Meliha Yetisgen (`melihay@uw.edu`) |
| Received date | TODO |
| Terms of use | TODO |
| `parameters.pkl` SHA-256 | TODO |
| `state_dict.pt` SHA-256 | TODO |
| Model version/training cohort | TODO |
| Allowed research or clinical uses | TODO |

Required files:

```text
data/external/uw_bionlp_ards/model/parameters.pkl
data/external/uw_bionlp_ards/model/state_dict.pt
```

Repository verification permits only these two configured files as untracked additions to the
pinned clone. Any tracked edit or other untracked file still blocks execution.

Do not substitute another model or infer the class mapping when these artifacts are unavailable.
