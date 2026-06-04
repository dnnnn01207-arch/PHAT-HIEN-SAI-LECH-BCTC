import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import openpyxl


st.set_page_config(page_title="Hệ thống Chẩn đoán Gian lận Tài chính", layout="wide")


class FraudDetectionApp:
    def __init__(self, model_path: str = "rf_model.pkl", train_data_path: str = "train_data.csv"):
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.model = None
        self.features = [
            'Total_Assets', 'Total_Liabilities', 'Revenue', 'Operating_Expenses', 
            'Net_Income', 'Cash_Flow_Operating', 'Cash_Flow_Investing', 
            'Cash_Flow_Financing', 'Current_Ratio', 'Debt_to_Equity', 
            'Gross_Margin', 'Return_on_Assets', 'Return_on_Equity'
        ]

    def read_file_with_encoding_handling(self, file_obj, file_extension):
        """Đọc file CSV hoặc XLSX với xử lý lỗi encoding"""
        if file_extension.lower() == 'xlsx':
            # Xử lý file Excel
            df = pd.read_excel(file_obj, engine='openpyxl')
            return df
        else:
            # Xử lý file CSV
            try:
                df = pd.read_csv(file_obj, encoding='utf-8')
                return df
            except UnicodeDecodeError:
                file_obj.seek(0)
                df = pd.read_csv(file_obj, encoding='latin-1')
                return df

    def load_or_train_model(self):
        """Kiểm tra file model pkl, nếu có thì load, nếu không thì train và save"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return self.model
        
        # Nếu chưa có model, đọc train_data.csv và train
        try:
            df = pd.read_csv(self.train_data_path)
            X = df[self.features]
            y = df['Financial_Status']
            
            # Khởi tạo và train mô hình
            rf = RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced',
                max_depth=12,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X, y)
            
            # Lưu model vào file
            joblib.dump(rf, self.model_path)
            self.model = rf
            return self.model
        except Exception as e:
            st.error(f"Lỗi khi tải hoặc huấn luyện mô hình: {e}")
            return None

    def run(self):
        st.title("🛡️ Hệ thống Phát hiện Gian lận Báo cáo Tài chính")
        st.markdown("---")
        
        # Sidebar: Thông tin hệ thống
        with st.sidebar:
            st.header("ℹ️ Thông tin hệ thống")
            st.info("✓ Hệ thống chạy **Offline** - Không yêu cầu API Key")
            st.markdown("---")
            st.write("**13 chỉ số tài chính được sử dụng:**")
            for feature in self.features:
                st.caption(f"• {feature}")
        
        # Tải model
        with st.spinner("Đang tải/huấn luyện mô hình..."):
            self.model = self.load_or_train_model()
        
        if self.model is None:
            st.error("Không thể tải hoặc huấn luyện mô hình. Vui lòng kiểm tra file train_data.csv")
            return
        
        st.success("✅ Mô hình sẵn sàng")
        st.markdown("---")
        
        # Phần chính: Upload file CSV hoặc XLSX
        st.header("📁 Tải lên dữ liệu tài chính")
        uploaded_file = st.file_uploader("Kéo/Thả file CSV hoặc XLSX hoặc chọn từ máy tính", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                # Lấy phần mở rộng file
                file_extension = uploaded_file.name.split('.')[-1]
                # Đọc file với xử lý encoding
                df = self.read_file_with_encoding_handling(uploaded_file, file_extension)
                
                # Kiểm tra các cột cần thiết
                missing_cols = [col for col in self.features if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ File CSV thiếu các cột: {', '.join(missing_cols)}")
                    return
                
                # Chọn công ty / dòng dữ liệu để phân tích
                st.markdown("---")
                st.subheader("🏢 Chọn công ty để phân tích")
                
                row_options = [f"Dòng {i+1}" for i in range(len(df))]
                selected_index = st.selectbox("Chọn công ty:", options=range(len(df)), format_func=lambda x: row_options[x])
                
                # Lấy dòng dữ liệu đã chọn
                selected_row = df[self.features].iloc[selected_index:selected_index+1]
                
                # Dự đoán
                prediction = self.model.predict(selected_row)[0]
                proba = self.model.predict_proba(selected_row)[0]
                
                # Hiển thị kết quả bằng st.columns
                st.markdown("---")
                st.header(f"📊 Kết quả chẩn đoán ({row_options[selected_index]})")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 0:
                        st.success("✅ **Bình thường**")
                        st.metric("Kết luận", "Class 0", delta="An toàn")
                    elif prediction == 1:
                        st.warning("⚠️ **Có dấu hiệu gian lận**")
                        st.metric("Kết luận", "Class 1", delta="Cảnh báo")
                    else:
                        st.error("🚨 **Rủi ro cao - Gian lận nghiêm trọng**")
                        st.metric("Kết luận", "Class 2", delta="Nguy hiểm")
                
                with col2:
                    st.metric("Xác suất Class 0 (Bình thường)", f"{proba[0]:.2%}")
                
                with col3:
                    st.metric("Xác suất Class 1 (Nghi vấn)", f"{proba[1]:.2%}")
                
                # Hiển thị xác suất chi tiết
                st.markdown("---")
                st.subheader("📈 Phân tích xác suất chi tiết")
                prob_df = pd.DataFrame({
                    "Loại": ["Class 0 - Bình thường", "Class 1 - Nghi vấn", "Class 2 - Rủi ro cao"],
                    "Xác suất": proba,
                    "Phần trăm": [f"{p:.2%}" for p in proba]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Hiển thị dữ liệu được sử dụng
                st.markdown("---")
                st.subheader(f"📋 Dữ liệu nhập vào ({row_options[selected_index]})")
                st.dataframe(selected_row, use_container_width=True, hide_index=True)
                
            except UnicodeDecodeError:
                st.error("❌ Lỗi encoding file. Vui lòng kiểm tra file CSV (UTF-8 hoặc Latin-1) hoặc file XLSX")
            except KeyError as e:
                st.error(f"❌ File CSV thiếu cột: {e}")
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý file: {e}")
        else:
            st.info("👆 Vui lòng tải lên file CSV hoặc XLSX để bắt đầu chẩn đoán")


if __name__ == "__main__":
    app = FraudDetectionApp(model_path="rf_model.pkl", train_data_path="train_data.csv")
    app.run()
