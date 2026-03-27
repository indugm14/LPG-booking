
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
import plotly.graph_objects as go

st.title("LPG Booking Management Dashboard")

df=pd.read_csv("final_dataset.csv")

st.header("Descriptive Analysis")
st.plotly_chart(px.histogram(df,x="Booking_Method",color="Complaint"))
st.plotly_chart(px.box(df,x="City_Tier",y="Delivery_Days"))
st.plotly_chart(px.scatter(df,x="Delivery_Days",y="Overall_Satisfaction",color="Complaint"))

st.header("Predictive Model")
df_enc=df.copy()
le=LabelEncoder()
for col in df_enc.columns:
    if df_enc[col].dtype=="object":
        df_enc[col]=le.fit_transform(df_enc[col])

X=df_enc.drop("Complaint",axis=1)
y=df_enc["Complaint"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=RandomForestClassifier()
model.fit(X_train,y_train)

pred=model.predict(X_test)
prob=model.predict_proba(X_test)[:,1]

st.write("Accuracy",accuracy_score(y_test,pred))
st.write("Precision",precision_score(y_test,pred))
st.write("Recall",recall_score(y_test,pred))
st.write("F1",f1_score(y_test,pred))

fpr,tpr,_=roc_curve(y_test,prob)
fig=go.Figure()
fig.add_trace(go.Scatter(x=fpr,y=tpr))
fig.add_trace(go.Scatter(x=[0,1],y=[0,1]))
st.plotly_chart(fig)
