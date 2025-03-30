import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

st.set_page_config(page_title="投资情绪 vs 收益分析", layout="wide")

st.title("📈 投资情绪与收益关系分析 App")

# --- 情绪关键词词典 ---
keywords = {
    "FOMO": ["错过", "抢涨", "大家都买", "涨疯", "疯涨", "朋友圈", "别人在买"],
    "焦虑": ["担心", "不确定", "焦虑", "犹豫", "波动", "紧张"],
    "兴奋": ["赚翻", "暴涨", "暴富", "兴奋", "要飞了", "机会难得"],
    "恐惧": ["清仓", "跳水", "完蛋", "血亏", "不敢看", "暴跌"]
}

def detect_emotion(text):
    for emo, words in keywords.items():
        for word in words:
            if re.search(word, text):
                return emo
    return "冷静"  # 默认情绪

# --- 数据输入区 ---
st.sidebar.header("输入你的每日投资记录")
date = st.sidebar.date_input("日期", value=datetime.date.today())
asset = st.sidebar.text_input("投资标的", "特斯拉")
reason = st.sidebar.text_area("原因")
system_emotion = detect_emotion(reason)
st.sidebar.markdown(f"🔍 系统识别情绪：**{system_emotion}**")
emotion = st.sidebar.selectbox("人工情绪标注（可选）", ["冷静", "焦虑", "兴奋", "FOMO", "恐惧"], index=["冷静", "焦虑", "兴奋", "FOMO", "恐惧"].index(system_emotion))
action = st.sidebar.selectbox("操作", ["买入", "卖出", "观望"])
profit = st.sidebar.number_input("当天盈亏（+为盈利，-为亏损）", value=0.0, step=0.1)

if st.sidebar.button("📥 保存记录"):
    new_row = pd.DataFrame({
        "日期": [date],
        "标的": [asset],
        "情绪": [emotion],
        "操作": [action],
        "盈亏": [profit],
        "原因": [reason],
        "系统识别情绪": [system_emotion]
    })
    try:
        history = pd.read_csv("investment_emotion_log.csv")
        updated = pd.concat([history, new_row], ignore_index=True)
    except FileNotFoundError:
        updated = new_row
    updated.to_csv("investment_emotion_log.csv", index=False)
    st.success("✅ 数据已保存")

# --- 可视化部分 ---
st.subheader("📊 情绪 vs 盈亏 可视化")

try:
    data = pd.read_csv("investment_emotion_log.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 不同情绪下的平均盈亏")
        avg_profit = data.groupby("情绪")["盈亏"].mean().sort_values()
        fig1, ax1 = plt.subplots()
        avg_profit.plot(kind="barh", ax=ax1)
        ax1.set_xlabel("平均盈亏")
        ax1.set_title("情绪与平均收益")
        st.pyplot(fig1)

    with col2:
        st.markdown("#### 情绪与盈亏散点图")
        fig2, ax2 = plt.subplots()
        sns.stripplot(x="情绪", y="盈亏", data=data, jitter=True, ax=ax2)
        ax2.axhline(0, color='gray', linestyle='--')
        st.pyplot(fig2)

    # 趋势图
    st.markdown("#### 收益与情绪趋势")
    data_sorted = data.sort_values("日期")
    fig3, ax3 = plt.subplots()
    for emo in data_sorted["情绪"].unique():
        subset = data_sorted[data_sorted["情绪"] == emo]
        ax3.plot(subset["日期"], subset["盈亏"], label=emo)
    ax3.legend()
    ax3.set_title("不同情绪下的收益走势")
    st.pyplot(fig3)

    # --- 机器学习部分 ---
    st.markdown("### 🤖 高风险情绪分析 (机器学习模型预测)")
    df = data.copy()
    df["高风险"] = df["盈亏"] < 0  # 将亏损视为高风险
    le_emotion = LabelEncoder()
    le_action = LabelEncoder()
    X = pd.DataFrame({
        "情绪编码": le_emotion.fit_transform(df["情绪"]),
        "操作编码": le_action.fit_transform(df["操作"])
    })
    y = df["高风险"].astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    st.text(classification_report(y, y_pred, target_names=["低风险", "高风险"]))

    feature_importance = model.feature_importances_
    st.markdown("#### 🎯 影响高风险的因素 (特征重要性)")
    fig4, ax4 = plt.subplots()
    ax4.bar(["情绪", "操作"], feature_importance)
    ax4.set_title("特征重要性分析")
    st.pyplot(fig4)

except FileNotFoundError:
    st.warning("还没有任何记录，快去左侧添加吧！")
