# Bài 3 — ResNet-18 trên VinaFood21

Huấn luyện ResNet-18 từ đầu cho bài toán phân loại 21 lớp VinaFood21. So sánh
hai chế độ:

- `noaug`: không dùng data augmentation;
- `aug`: dùng data augmentation.

Best checkpoint được chọn theo **validation macro-F1**; test không được dùng để
chọn epoch.

## Cấu trúc

```text
Bai3/
├── run_bai3.ipynb
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

| Thuộc tính | No augmentation | Augmentation |
|---|---:|---:|
| Epoch | 20 | 20 |
| Ảnh | 96×96 | 96×96 |
| Batch size | 32 | 32 |
| Optimizer | Adam | Adam |
| Learning rate | `1e-3` | `1e-3` |
| Tổng tham số | 11.187.285 | 11.187.285 |

Validation được tách phân tầng 15%, seed 42. Validation/test không dùng
augmentation.

## Kết quả

| Chế độ | Best epoch | Val macro-F1 | Test accuracy | Precision-macro | Recall-macro | Test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| No augmentation | 20 | 0.4959 | 0.4244 | 0.4255 | 0.4095 | 0.4090 |
| Augmentation | **20** | **0.5236** | **0.5049** | **0.5152** | **0.4986** | **0.4923** |

## Chạy bằng terminal

```powershell
pip install -r requirements.txt
python train.py --data_root "DUONG_DAN_VINAFOOD21" --variant both --epochs 20
python evaluate.py --data_root "DUONG_DAN_VINAFOOD21" --variant noaug
python evaluate.py --data_root "DUONG_DAN_VINAFOOD21" --variant aug
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
