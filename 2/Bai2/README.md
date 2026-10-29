# Bài 2: GoogLeNet trên VinaFood21

Phân loại 21 lớp VinaFood21 bằng GoogLeNet rút gọn gồm 4 Inception block,
global average pooling và classifier 21 lớp.

Best checkpoint được chọn theo **validation macro-F1**; test không được dùng để
chọn epoch.

## Cấu trúc

```text
Bai2/
├── run_b2.ipynb
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

| Thuộc tính | Giá trị |
|---|---:|
| Epoch | 20 |
| Ảnh | 64×64 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Tổng tham số | 179.877 |

Validation được tách phân tầng 15%, seed 42. Train có augmentation;
validation/test không augmentation.

## Kết quả

| Model | Best epoch | Val macro-F1 | Test accuracy | Precision-macro | Recall-macro | Test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| GoogLeNet | 20 | 0.4097 | 0.4181 | 0.4074 | 0.3930 | 0.3902 |

## Chạy bằng terminal

```powershell
pip install -r requirements.txt
python train.py --data_root "DUONG_DAN_VINAFOOD21" --epochs 20
python evaluate.py --data_root "DUONG_DAN_VINAFOOD21"
```

## File kết quả

`outputs/` chứa:

```text
Bai2_GoogLeNet_best.pt
Bai2_GoogLeNet_history.csv
Bai2_GoogLeNet_loss.png
Bai2_GoogLeNet_test.json
Bai2_GoogLeNet_per_class.csv
Bai2_GoogLeNet_confusion_matrix.png
```
