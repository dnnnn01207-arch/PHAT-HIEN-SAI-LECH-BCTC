import pandas as pd

# 1. Đọc dữ liệu
df = pd.read_csv('Financial Statement Anomaly Dataset.csv')

# 2. Xóa khoảng trắng thừa trong cột trạng thái (nếu có)
df['Financial_Status'] = df['Financial_Status'].str.strip()

# 3. Mã hóa Nhãn (Label Encoding) cho biến mục tiêu
# Quy ước: 0 là Bình thường, 1 là Dấu hiệu gian lận, 2 là Rủi ro cao
status_mapping = {
    'Normal': 0, 
    'Anomaly': 1, 
    'High Risk': 2
}
df['Financial_Status'] = df['Financial_Status'].map(status_mapping)

# 4. Tách dữ liệu thành Đặc trưng (X) và Nhãn (y)
X = df.drop(columns=['Financial_Status']) # Lấy toàn bộ 13 cột tài chính
y = df['Financial_Status']                # Lấy cột trạng thái làm mục tiêu để học

# Xác nhận kết quả
print("Kích thước dữ liệu X:", X.shape)
print("10 dòng đầu tiên của nhãn y:\n", y.head(10))

# Lưu dữ liệu đã làm sạch vào file mới (nếu cần)
cleaned_df = pd.concat([X, y], axis=1)
cleaned_df.to_csv('cleaned_financial_data.csv', index=False)