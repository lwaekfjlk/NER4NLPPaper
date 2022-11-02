python -m explainaboard_client.cli.evaluate_system \
  --username $EB_USERNAME \
  --api-key $EB_API_KEY \
  --task named-entity-recognition \
  --system-name anlp_haofeiy_sciner \
  --custom-dataset-file ./data/anlp_valid/anlp-sciner-valid-annotated.conll \
  --custom-dataset-file-type conll  \
  --system-output-file ./data/anlp_valid/anlp-sciner-valid-sentences-with-scirex.conll \
  --system-output-file-type conll \
  --source-language en \
