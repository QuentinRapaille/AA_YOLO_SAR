# AA-YOLO-SAR (Working Fork)

Ce depot est une version orientee usage SAR maritime de AA-YOLO (base YOLOv7), utilisee ici pour la detection de navires sur des jeux de donnees de type HRSID, SSDD et LS-SSDD.

Le projet conserve deux variantes de tete de detection:

- `cfg/training/AA-yolov7-tiny.yaml` avec `IDetect_AA` (AA-YOLO)
- `cfg/training/yolov7-tiny.yaml` avec `IDetect` (baseline YOLOv7-tiny)

Le comportement des losses est pilote dans `data/hyp.scratch.AA_yolo.yaml`, notamment:

- `loss_AA: 1`
- `loss_ota: 1`

## Environnement

Depuis la racine du projet:

```bash
uv sync
```

Alternative:

```bash
pip install -r requirements.txt
```

## Jeux de donnees utilises dans ce fork

Fichiers de config presents:

- `data/hrsid.yaml`
- `data/ssdd.yaml`
- `data/ls_ssdd_v1.yaml`

Sorties YOLO associees:

- `data/datasets/HRSID_YOLO/`
- `data/datasets/SSDD_YOLO/`
- `data/datasets/LS_SSDD_v1_YOLO/`

## Preparation des donnees

Scripts fournis dans la racine du depot:

- `convert_hrsid_coco_to_yolo.py`
- `prepare_ssdd.py`
- `prepare_ls_ssdd_v1.py`
- `prepare_sar_optionA.py` (pipeline generique SAR custom)

Exemples:

```bash
python convert_hrsid_coco_to_yolo.py
python prepare_ssdd.py
python prepare_ls_ssdd_v1.py
```

## Entrainement

AA-YOLO (tete anomaly-aware) sur HRSID:

```bash
python train.py \
  --workers 8 \
  --batch-size 16 \
  --data data/hrsid.yaml \
  --img-size 640 640 \
  --iou-thres 0.05 \
  --epochs 400 \
  --cfg cfg/training/AA-yolov7-tiny.yaml \
  --hyp data/hyp.scratch.AA_yolo.yaml \
  --name train_hrsid_optionA \
  --single-cls
```

Baseline YOLOv7-tiny sur HRSID:

```bash
python train.py \
  --workers 8 \
  --batch-size 16 \
  --data data/hrsid.yaml \
  --img-size 640 640 \
  --iou-thres 0.05 \
  --epochs 400 \
  --cfg cfg/training/yolov7-tiny.yaml \
  --hyp data/hyp.scratch.AA_yolo.yaml \
  --name train_hrsid_yolo_baseline \
  --single-cls
```

Le dossier de sortie est `runs/train/<name>/`.

## Evaluation

Exemple de test sur HRSID:

```bash
python test.py \
  --batch-size 16 \
  --data data/hrsid.yaml \
  --img-size 640 \
  --iou-thres 0.05 \
  --task test \
  --weights runs/train/train_hrsid_optionA/weights/best.pt \
  --name test_hrsid_optionA \
  --single-cls \
  --exist-ok
```

Les resultats sont ecrits dans `runs/test/<name>/` (`results.txt`, courbes PR/F1, etc.).

## SAR custom (Option A)

Le script `prepare_sar_optionA.py` permet de preparer un dataset SAR custom (normalisation percentile, export PNG 3 canaux, split train/val/test).

Exemple:

```bash
python prepare_sar_optionA.py \
  --images-dir /chemin/vers/images \
  --labels-dir /chemin/vers/labels_yolo \
  --dataset-name SAR_OPTION_A \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1
```

Ensuite, creer un fichier `.yaml` dedie dans `data/` pointant vers les `.txt` generes.

## Citation

Si vous utilisez AA-YOLO, merci de citer l'article associe (voir `CITATION.cff`):

- DOI: `10.1016/j.engappai.2026.114186`
- URL: <https://www.sciencedirect.com/science/article/pii/S0952197626004677>

## Remerciements

Ce travail est base sur YOLOv7: <https://github.com/WongKinYiu/yolov7>
test
