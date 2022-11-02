# 11711 HW02

Setting1 To run the SciNER finetuning experiment

```bash
sh run_sciner-finetune.sh
```

Setting2 To run the SciRex finetuning experiment

```bash
sh run_scirex-finetune.sh
```

Setting3 To run the SciNER finetuning based on SciRex finetuning checkpoint

```bash
sh run_sciner-continuous-finetune.sh
```

Setting4 To get the inference results based on the official test dataset

```bash
sh run_sciner-test-inference.sh
```

Setting5 To get the inference results based on the self-made validation dataset

```bash
sh run_sciner-valid-inference.sh
```

Setting6 To submit the inference result files to the ExplanaBoard to see the final result
```bash
sh explain_board_test_submission.sh
sh explain_board_valid_submission.sh
```

