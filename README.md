# American-Express-Competition-with-pyspark
Vấn đề chấm điểm tín dụng nằm trên tập dữ liệu lớn với pyspark. Dự đoán xem khách hàng có vỡ nợ trong tương lai hay không.

## 1. Nội dung
Đây là repo thực hiện bài tập lớn môn **Nhập môn khai phá tập dữ liệu lớn**, tập trung vào một số nội dung sau 
- Tìm hiều về các thuật toán K-means, Decision Tree và SVM, cách song song hóa các thuật toán này với tập dữ liệu lớn 
- Triển khai các thuật toán trên thông qua pyspark 
- Một số kỹ thuật xử lý giá trị bị thiếu 
- Một số hướng tiếp cận để giảm kích thước bộ dữ liệu ban đầu 
- Khai thác một số insight của dữ liệu (EDA) 

## 2. Dữ liệu
Một số lưu ý về dữ liệu:
- `dataset`: thư mục này chứa dữ liệu gốc được lấy từ đường link [https://www.kaggle.com/competitions/amex-default-prediction/data](https://www.kaggle.com/competitions/amex-default-prediction/data)
- `processedDataset`: thư mục này chứa dữ liệu sau khi đã được xử lý bằng cách thay đổi kiểu dữ liệu của mỗi trường từ `dataset`, được lưu dưới định dạng `parquet`, có thể được lấy từ [https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format), hoặc xử lý từ đầu qua hướng dẫn của báo cáo (Report).
- `optimalDataset`: thư mục này chứa dữ liệu sau khi đã được tiền làm sạch, EDA để đưa và mô hình huấn luyện, có thể được lấy qua `convert.py`

**Tuy nhiên, do các dữ liệu này đều có không gian lưu trữ quá lớn, nên các bạn có thể truy cập vào các thư mục dữ liệu trên tại [https://drive.google.com/drive/folders/1Ddcg9bcuI1nYEVPgI5V5wCyKnzBVFZt3?usp=sharing](https://drive.google.com/drive/folders/1Ddcg9bcuI1nYEVPgI5V5wCyKnzBVFZt3?usp=sharing)**

## 3. Chạy chương trình 
Để chạy chương trình, trước tiên cần đảm bảo có một số thư viện cần thiết như: numpy, pandas, pyspark, ... và một số thư viện khác. 
Sau đó, chuyền đường dẫn làm việc đến thư mục gốc `Mining Massive Dataset`

```buildoutcfg
cd path/to/Mining Massive Dataset
```

Để chuyển đồi dữ liệu từ `processedDataset` sang `optimalDataset`:
```buildoutcfg
python3 convert.py 
```

Để chạy chương trình huấn luyện các thuật toán:
```buildoutcfg
python3 models.py 
```

## 4. Hướng dẫn chi tiết 
Trong trường hợp không muốn chạy lại chương trình từ đầu, có thể xem qua thư mục `Tutorial` gồm 
- `Exploratory Data Analysis.ipynb`: Mô phỏng lại các quá trình, cũng như nêu kết quả từ việc khai phá dữ liệu tới chuyển đổi dữ liệu 
- `Modeling.ipynb`: Mô phỏng và lưu lại kết quả của môt lần chạy các thuật toán song song hóa qua pyspark 

## 5. Chú ý 
Ngoài ra, bạn đọc có thể xem qua báo cáo `Report` để hiểu hơn. 