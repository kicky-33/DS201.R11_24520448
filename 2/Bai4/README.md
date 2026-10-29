# Bài 4 — Fine-tuning pretrained ResNet-50

Phân loại 21 lớp VinaFood21 bằng pretrained `microsoft/resnet-50` từ Hugging
Face. So sánh hai chế độ:

- `feature`: đóng băng backbone, chỉ train classifier;
- `finetune`: train toàn bộ mô hình.

Best checkpoint được chọn theo **validation macro-F1**; test không được dùng để
chọn epoch.

## Cấu trúc

```text
Bai4/
├── run_b4.ipynb
├── data.py
├── engine.py
├── evaluate.py
├── model.py
├── train.py
├── requirements.txt
├── README.md
└── outputs/
```

## Cấu hình

| Thuộc tính | Feature | Fine-tune |
|---|---:|---:|
| Epoch | 20 | 20 |
| Ảnh | 224×224 | 224×224 |
| Batch size | 32 | 32 |
| Optimizer | Adam | Adam |
| Learning rate | `1e-3` | `1e-4` |
| Trainable params | 43.029 | 23.551.061 |

Validation được tách phân tầng 15%, seed 42. Train có augmentation;
validation/test không augmentation.

## Kết quả

| Chế độ | Best epoch | Val macro-F1 | Test accuracy | Precision-macro | Recall-macro | Test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Feature | 20 | 0.7797 | 0.7462 | 0.7601 | 0.7571 | 0.7539 |
| Fine-tune | **19** | **0.8805** | **0.8608** | **0.8635** | **0.8640** | **0.8621** |

## Chạy bằng terminal

```powershell
pip install -r requirements.txt
python train.py --data_root "DUONG_DAN_VINAFOOD21" --mode both --epochs 20
python evaluate.py --data_root "DUONG_DAN_VINAFOOD21" --mode feature
python evaluate.py --data_root "DUONG_DAN_VINAFOOD21" --mode finetune
```

## File kết quả

`outputs/` chứa cho mỗi chế độ:

```text
*_best.pt
*_history.csv
*_loss.png
*_test.json
*_per_class.csv
*_confusion_matrix.png
```


