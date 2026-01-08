import pandas as pd
import openai
df=pd.read_csv("Dataset/train.csv")
row = df[df['id']==46].iloc[0]

