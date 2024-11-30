import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_functions import predict_scores, plot_student_prediction
from sklearn.linear_model import LinearRegression
import mplcursors

# 設置頁面
st.set_page_config(page_title="成績數據分析系統", layout="wide")

# 主標題
st.title("成績數據分析 📊")
use_demo_data = st.checkbox("使用示範數據")

def generate_demo_data():
    """生成示範數據"""
    np.random.seed(42)
    data = {
        '*School Year': ['112'] * 100,
        '*Class Level': [str(x) for x in np.random.choice([1,2,3], 100)],
        '*Class': [f'{cl}{num}' for cl, num in zip(np.random.choice(['甲','乙','丙'], 100), np.random.choice([1,2], 100))],
        '*Class Number': np.random.randint(1, 40, 100),
        '*Student Name': [f'學生{i}' for i in range(1, 101)],
    }
    
    # 生成考試成績
    for term in [1, 2]:
        for assessment in [1, 2]:
            for lang in ['C', 'E']:
                col_name = f'T{term}A{assessment}_Math_{lang}_Score'
                data[col_name] = np.random.normal(75, 15, 100)
    
    return pd.DataFrame(data)

def process_score_columns(df):
    """處理成績欄位"""
    required_columns = ['*School Year', '*Class Level', '*Class', '*Class Number', '*Student Name']
    
    # 檢查必要欄位是否存在
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"缺少必要的欄位: {', '.join(missing_columns)}")
        return None
    
    # 找出所有包含 'Score' 的欄位
    score_columns = [col for col in df.columns if 'Score' in col]
    
    # 重組數據
    melted_df = pd.melt(
        df,
        id_vars=required_columns,
        value_vars=score_columns,
        var_name='Exam',
        value_name='Score'
    )
    
    # 提取考試資訊
    melted_df['Term_Assessment'] = 'T' + melted_df['Exam'].str.extract(r'T(\d+)A(\d+)')[0] + \
                                  'A' + melted_df['Exam'].str.extract(r'T(\d+)A(\d+)')[1]
    melted_df['Language'] = melted_df['Exam'].str.extract(r'_([CE])_')[0].map({'C': '中文卷', 'E': '英文卷'})
    
    # 將分數轉換為數字類型，並排除無效數據
    melted_df['Score'] = pd.to_numeric(melted_df['Score'], errors='coerce')
    melted_df = melted_df.dropna(subset=['Score'])
    
    return melted_df

def plot_predictions(predictions, prediction_type, school_year, subject_name):
    """繪製預測圖表"""
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    if prediction_type == "班級預測":
        plt.title(f"{school_year}學年度 {subject_name} {prediction_type}")
        for class_pred_key, pred_data in predictions['class_predictions'].items():
            class_name, exam_type = class_pred_key.rsplit('_', 1)
            historical_data = pred_data['historical_data']
            term_labels = pred_data['term_labels']
            
            x_values = range(len(historical_data))
            line = plt.plot(x_values, list(historical_data.values()), 
                    marker='o', 
                    label=f'{class_name}({exam_type})')
            
            # 添加數據標籤
            for x, y in zip(x_values, list(historical_data.values())):
                plt.annotate(f'{y:.1f}', 
                           (x, y),
                           textcoords="offset points",
                           xytext=(0,10),
                           ha='center')
            
            pred_line = plt.plot(len(x_values), pred_data['predicted_score'], 
                    marker='*', 
                    markersize=15,
                    label=f'{class_name}({exam_type}) 預測')
            
            # 添加預測值標籤
            plt.annotate(f'{pred_data["predicted_score"]:.1f}', 
                       (len(x_values), pred_data['predicted_score']),
                       textcoords="offset points",
                       xytext=(0,10),
                       ha='center')
            
            plt.xticks(list(x_values) + [len(x_values)], 
                      term_labels + ['預測'],
                      rotation=45)
    
    elif prediction_type == "年級預測":
        plt.title(f"{school_year}學年度 {subject_name} {prediction_type}")
        for level, pred_data in predictions['level_predictions'].items():
            historical_data = pred_data['historical_data']
            term_labels = pred_data['term_labels']
            
            x_values = range(len(historical_data))
            line = plt.plot(x_values, list(historical_data.values()), 
                    marker='o', 
                    label=f'{level}年級')
            
            # 添加數據標籤
            for x, y in zip(x_values, list(historical_data.values())):
                plt.annotate(f'{y:.1f}', 
                           (x, y),
                           textcoords="offset points",
                           xytext=(0,10),
                           ha='center')
            
            pred_line = plt.plot(len(x_values), pred_data['predicted_score'], 
                    marker='*', 
                    markersize=15,
                    label=f'{level}年級預測')
            
            # 添加預測值標籤
            plt.annotate(f'{pred_data["predicted_score"]:.1f}', 
                       (len(x_values), pred_data['predicted_score']),
                       textcoords="offset points",
                       xytext=(0,10),
                       ha='center')
            
            plt.xticks(list(x_values) + [len(x_values)], 
                      term_labels + ['預測'],
                      rotation=45)
    
    plt.xlabel("考試次序")
    plt.ylabel("平均分")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 添加互動提示
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(
            f'考試：{term_labels[int(sel.target[0])] if sel.target[0] < len(term_labels) else "預測"}\n'
            f'平均分：{sel.target[1]:.1f}'))
    
    st.pyplot(plt)
    plt.close()

def plot_student_prediction(student_exam_data, selected_student, exam_type):
    """繪製學生預測圖表"""
    plt.figure(figsize=(10, 6))
    plt.title(f"{selected_student} 的 {exam_type} 成績預測")
    
    # 假設 student_exam_data 包含 'Term_Assessment' 和 'Score' 欄位
    plt.plot(student_exam_data['Term_Assessment'], student_exam_data['Score'], marker='o', label='歷史成績')
    
    # 添加預測邏輯（這裡可以根據您的需求進行調整）
    # 例如，這裡可以使用線性回歸進行預測
    # 這裡僅為示範，您需要根據實際需求進行實現
    predicted_score = student_exam_data['Score'].mean() + 5  # 假設預測分數
    plt.axhline(y=predicted_score, color='r', linestyle='--', label='預測分數')
    
    plt.xlabel("考試次序")
    plt.ylabel("分數")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

# 修改檔案上傳部分
if use_demo_data:
    df = generate_demo_data()
    processed_df = process_score_columns(df)
    st.info("目前使用示範數據，您可以取消勾選上方的「使用示範數據」來上傳自己的資料")
else:
    uploaded_file = st.file_uploader("上傳成績檔案（Excel格式）", type=['xlsx', 'xls'])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            processed_df = process_score_columns(df)
        except Exception as e:
            st.error(f"處理檔案時發���錯誤：{str(e)}")
    else:
        st.info("請上傳成績檔案開始分析，或使用示範數據")
        processed_df = None

if processed_df is not None:
    # 提取科目名稱
    subject_name = processed_df['Exam'].iloc[0].split('_')[1]
    # 提取學年
    school_year = processed_df['*School Year'].iloc[0]
    
    st.subheader(f"{school_year}學年度 {subject_name} 成績預測分析")
    
    # 進行預測分析
    predictions = predict_scores(processed_df)
    
    # 顯示不同類型的預測
    tabs = st.tabs(["班級預測", "年級預測", "個人預測"])
    
    with tabs[0]:
        st.subheader("📊 班級預測")
        if predictions and predictions['class_predictions']:
            plot_predictions(predictions, "班級預測", school_year, subject_name)
        else:
            st.warning("沒有足夠的數據進行班級預測")
    
    with tabs[1]:
        st.subheader("📈 年級預測")
        if predictions and predictions['level_predictions']:
            plot_predictions(predictions, "年級預測", school_year, subject_name)
        else:
            st.warning("沒有足夠的數據進行年級預測")
    
    with tabs[2]:
        st.subheader("👤 個人預測")
        # 選擇班級
        class_options = sorted(processed_df['*Class'].unique())
        selected_class = st.selectbox("選擇班別", class_options)
        
        if selected_class:
            # 過濾該班級的學生
            class_students = processed_df[
                (processed_df['*Class'] == selected_class) & 
                (processed_df['Score'].notna())
            ]['*Student Name'].unique()
            
            # 選擇學生
            selected_student = st.selectbox("選擇學生", sorted(class_students))
            
            if selected_student:
                # 獲取該學生的考卷類型
                student_data = processed_df[processed_df['*Student Name'] == selected_student]
                available_exam_types = student_data['Language'].unique()
                
                if len(available_exam_types) > 0:
                    exam_type = st.radio(
                        "選擇考卷類型",
                        available_exam_types,
                        horizontal=True
                    )
                    
                    # 過濾並顯示該學生特定考卷類型的成績趨勢
                    student_exam_data = student_data[
                        student_data['Language'] == exam_type
                    ].sort_values('Term_Assessment')
                    
                    if len(student_exam_data) >= 2:
                        plot_student_prediction(student_exam_data, selected_student, exam_type)
                    else:
                        st.warning(f"該學生的{exam_type}考試次數不足，無法進行預測")

with st.expander("使用說明"):
    st.markdown("""
    ### 如何使用本系統：
    1. 選擇「使用示範數據」或上傳您的成績檔案
    2. 系統會自動分析並顯示：
        - 班級預測
        - 年級預測
        - 個人預測
    3. 您可以切換不同分析頁面查看詳細資訊
    
    ### 資料格式要求：
    - Excel檔案必須包含以下欄位：
        - *School Year（學年度）
        - *Class Level（年級）
        - *Class（班級）
        - *Class Number（座號）
        - *Student Name（學生姓名）
    - 成績欄位格式：T{term}A{assessment}_{Subject}_{Language}_Score
    """)