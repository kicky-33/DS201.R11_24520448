# Bài 1 — LeNet-5 trên VinaFood21

Phân loại 21 lớp VinaFood21 bằng LeNet-5 điều chỉnh cho ảnh RGB 64×64. So sánh
hai loại pooling:

- `avg`: Average Pooling;
- `max`: Max Pooling.

Best checkpoint được chọn theo **validation macro-F1**; test không được dùng để
chọn epoch.

## Cấu trúc

```text
Bai1/
├── run_b1.ipynb
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

| Thuộc tính | AvgPool | MaxPool |
|---|---:|---:|
| Epoch | 20 | 20 |
| Ảnh | 64×64 | 64×64 |
| Batch size | 32 | 32 |
| Optimizer | Adam | Adam |
| Learning rate | `1e-3` | `1e-3` |
| Tổng tham số | 339.421 | 339.421 |

Validation được tách phân tầng 15%, seed 42. Train có augmentation;
validation/test không augmentation.

## Kết quả

| Chế độ | Best epoch | Val macro-F1 | Test accuracy | Precision-macro | Recall-macro | Test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| AvgPool | 19 | 0.3315 | 0.3031 | 0.2929 | 0.2897 | 0.2800 |
| MaxPool | **13** | **0.3388** | **0.3239** | **0.3235** | **0.3047** | **0.2967** |

## Chạy bằng terminal

```powershell
pip install -r requirements.txt
python train.py --data_root "DUONG_DAN_VINAFOOD21" --pool both --epochs 20
python evaluate.py --data_root "DUONG_DAN_VINAFOOD21"
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
