import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def analyze_performance(df):
    """基本成績分析"""
    results = {
        'by_school_year': df.groupby('*School Year')['Score'].agg(['mean', 'std']).round(2),
        'by_class_level': df.groupby('*Class Level')['Score'].agg(['mean', 'std']).round(2),
        'by_class': df.groupby(['*Class Level', '*Class'])['Score'].agg(['mean', 'std']).round(2)
    }
    return results

def analyze_cross_year_performance(df):
    """跨年度分析"""
    try:
        results = {
            'summary': pd.DataFrame(),
            'trends': {},
            'comparisons': {}
        }
        
        # 確保數據欄位存在
        if 'Year' not in df.columns:
            df['Year'] = pd.to_datetime(df['Exam Date']).dt.year
            
        # 計算年度統計
        yearly_stats = df.groupby('Year').agg({
            'Score': ['mean', 'median', 'std', 'count']
        }).round(2)
        
        results['summary'] = yearly_stats
        return results
    except Exception as e:
        print(f"跨年度分析錯誤: {str(e)}")
        return None

def analyze_version_comparison(df):
    """版別比較分析"""
    try:
        results = {
            'summary': pd.DataFrame(),
            'differences': {}
        }
        
        if 'Version' in df.columns:
            version_stats = df.groupby('Version').agg({
                'Score': ['mean', 'median', 'std', 'count']
            }).round(2)
            
            results['summary'] = version_stats
        return results
    except Exception as e:
        print(f"版別比較分析錯誤: {str(e)}")
        return None

def analyze_overall_performance(df):
    """整體表現分析"""
    try:
        results = {
            'summary': {},
            'distributions': {}
        }
        
        # 基本統計
        results['summary'] = {
            '平均分': df['Score'].mean(),
            '中位數': df['Score'].median(),
            '標準差': df['Score'].std(),
            '及格率': (df['Score'] >= 50).mean() * 100,
            '優良率': (df['Score'] >= 70).mean() * 100
        }
        
        return results
    except Exception as e:
        print(f"整體表現分析錯誤: {str(e)}")
        return None

def analyze_learning_gaps(df):
    """學習差異分析"""
    try:
        results = {
            'gaps': {},
            'groups': {}
        }
        
        # 班級差異分析
        if '*Class' in df.columns:
            class_stats = df.groupby('*Class')['Score'].agg([
                '平均分',
                '標準差',
                '最高分',
                '最低分'
            ]).round(2)
            
            results['gaps']['class'] = class_stats
            
        return results
    except Exception as e:
        print(f"學習差異分析錯誤: {str(e)}")
        return None

def analyze_progress(df):
    """進步程度分析"""
    try:
        results = {
            'individual_progress': {},
            'class_progress': {}
        }
        
        # 確保有考試日期
        if 'Exam Date' in df.columns and 'Student ID' in df.columns:
            df_sorted = df.sort_values(['Student ID', 'Exam Date'])
            df_sorted['Score_Change'] = df_sorted.groupby('Student ID')['Score'].diff()
            
            results['individual_progress'] = {
                '進步人數': (df_sorted['Score_Change'] > 0).sum(),
                '退步人數': (df_sorted['Score_Change'] < 0).sum(),
                '平均進步分數': df_sorted['Score_Change'].mean()
            }
            
        return results
    except Exception as e:
        print(f"進步程度分析錯誤: {str(e)}")
        return None

def predict_scores(df):
    """預測分析"""
    results = {
        'class_predictions': {},
        'level_predictions': {},
        'individual_predictions': {},
        'explanation': {}
    }
    
    # 1. 班級C/E卷預測
    for exam_type in ['中文卷', '英文卷']:
        for class_name in df['*Class'].unique():
            class_data = df[
                (df['*Class'] == class_name) & 
                (df['Language'] == exam_type)
            ]
            # 計算每次考試的平均分
            class_means = class_data.groupby('Term_Assessment')['Score'].mean()
            
            if len(class_means) >= 2:
                X = np.array(range(len(class_means))).reshape(-1, 1)
                y = class_means.values
                model = LinearRegression()
                model.fit(X, y)
                
                next_term = len(class_means)
                prediction = model.predict([[next_term]])[0]
                r2_score = model.score(X, y)
                
                results['class_predictions'][f"{class_name}_{exam_type}"] = {
                    'predicted_score': round(prediction, 1),
                    'confidence': round(r2_score * 100, 1),
                    'trend': 'upward' if model.coef_[0] > 0 else 'downward',
                    'historical_data': class_means.to_dict(),
                    'next_term': next_term,
                    'term_labels': list(class_means.index)
                }
    
    # 2. 年級整體預測（不分卷別）
    for level in df['*Class Level'].unique():
        level_data = df[df['*Class Level'] == level]
        level_means = level_data.groupby('Term_Assessment')['Score'].mean()
        
        if len(level_means) >= 2:
            X = np.array(range(len(level_means))).reshape(-1, 1)
            y = level_means.values
            model = LinearRegression()
            model.fit(X, y)
            
            next_term = len(level_means)
            prediction = model.predict([[next_term]])[0]
            r2_score = model.score(X, y)
            
            results['level_predictions'][level] = {
                'predicted_score': round(prediction, 1),
                'confidence': round(r2_score * 100, 1),
                'trend': 'upward' if model.coef_[0] > 0 else 'downward',
                'historical_data': level_means.to_dict(),
                'next_term': next_term,
                'term_labels': list(level_means.index)
            }
    
    return results

def analyze_by_year_group(df, selected_years):
    """依年度分組分析"""
    results = {
        'summary': pd.DataFrame(),
        'comparisons': {},
        'student_tracking': {}
    }
    
    # 只分析選定的年度
    df_filtered = df[df['Year'].isin(selected_years)]
    
    # 基本統計
    yearly_stats = df_filtered.groupby('Year').agg({
        'Score': ['mean', 'median', 'std', 'count'],
        'Student ID': 'nunique'
    }).round(2)
    
    # 追蹤學生在不同年度的表現
    common_students = set.intersection(*[
        set(df[df['Year'] == year]['Student ID']) 
        for year in selected_years
    ])
    
    if common_students:
        student_progress = df_filtered[
            df_filtered['Student ID'].isin(common_students)
        ].pivot_table(
            index='Student ID',
            columns='Year',
            values='Score'
        )
        
        results['student_tracking'] = student_progress
    
    results['summary'] = yearly_stats
    return results 

def plot_student_prediction(student_exam_data, selected_student, exam_type):
    """繪製學生預測圖表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左側圖：成績趨勢
    ax1.set_title(f"{selected_student} 的 {exam_type} 成績趨勢")
    ax1.plot(student_exam_data['Term_Assessment'], student_exam_data['Score'], 
            marker='o', label='歷史成績', color='blue')
    
    # 計算統計數據
    mean_score = student_exam_data['Score'].mean()
    median_score = student_exam_data['Score'].median()
    std_score = student_exam_data['Score'].std()
    q1 = student_exam_data['Score'].quantile(0.25)
    q3 = student_exam_data['Score'].quantile(0.75)
    min_score = student_exam_data['Score'].min()
    max_score = student_exam_data['Score'].max()
    
    # 添加預測
    X = np.arange(len(student_exam_data)).reshape(-1, 1)
    y = student_exam_data['Score'].values
    model = LinearRegression()
    model.fit(X, y)
    confidence = model.score(X, y)
    next_score = model.predict([[len(student_exam_data)]])[0]
    
    # 繪製預測點
    ax1.plot(len(student_exam_data), next_score, 
            marker='*', markersize=15, color='red', label='預測分數')
    
    # 添加統計資訊文字
    stats_text = (
        f'統計資訊：\n'
        f'平均數：{mean_score:.1f}\n'
        f'中位數：{median_score:.1f}\n'
        f'標準差：{std_score:.1f}\n'
        f'最大值：{max_score:.1f}\n'
        f'最小值：{min_score:.1f}\n'
        f'上四分位：{q3:.1f}\n'
        f'下四分位：{q1:.1f}\n'
        f'預測信心：{confidence*100:.1f}%'
    )
    ax1.text(1.02, 0.5, stats_text, transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 右側圖：箱形圖
    ax2.set_title(f"{selected_student} 的 {exam_type} 成績分布")
    ax2.boxplot(student_exam_data['Score'], 
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"))
    
    plt.tight_layout()
    return fig