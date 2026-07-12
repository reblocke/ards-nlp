CREATE OR REPLACE TABLE `{{ project_id }}.{{ bq_dataset }}.mimic_cxr_jpg_labels_normalized` AS
SELECT
  COALESCE(cx.subject_id, nb.subject_id) AS subject_id,
  COALESCE(cx.study_id, nb.study_id) AS study_id,

  SAFE_CAST(cx.lung_opacity AS INT64) AS chexpert_lung_opacity,
  SAFE_CAST(cx.edema AS INT64) AS chexpert_edema,
  SAFE_CAST(cx.consolidation AS INT64) AS chexpert_consolidation,
  SAFE_CAST(cx.atelectasis AS INT64) AS chexpert_atelectasis,
  SAFE_CAST(cx.pleural_effusion AS INT64) AS chexpert_pleural_effusion,
  SAFE_CAST(cx.pneumonia AS INT64) AS chexpert_pneumonia,

  SAFE_CAST(nb.lung_opacity AS INT64) AS negbio_lung_opacity,
  SAFE_CAST(nb.edema AS INT64) AS negbio_edema,
  SAFE_CAST(nb.consolidation AS INT64) AS negbio_consolidation,
  SAFE_CAST(nb.atelectasis AS INT64) AS negbio_atelectasis,
  SAFE_CAST(nb.pleural_effusion AS INT64) AS negbio_pleural_effusion,
  SAFE_CAST(nb.pneumonia AS INT64) AS negbio_pneumonia
FROM `{{ project_id }}.{{ bq_dataset }}.stg_mimic_cxr_jpg_chexpert` AS cx
FULL OUTER JOIN `{{ project_id }}.{{ bq_dataset }}.stg_mimic_cxr_jpg_negbio` AS nb
  USING (subject_id, study_id);
