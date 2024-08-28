import pandas as pd
import plotly.express as px


def plot_fig(df):
    # plotting some dfs in plotly 
    fig = px.scatter(df, title="reconf benchmark datasets")
    fig.show()
    return

df1 = pd.read_csv("../datasets/ds1/ds1n.csv")
#df2 = pd.read_csv("../datasets/ds1/ds1l.csv")
#df3 = pd.read_csv("../datasets/ds1/ds1lc.csv")

df_list = [df1]#, df2, df3]

plot_fig(df1)

channels_of_interest = ["time", "level", "m_flow", "v_flow", 
    "fluidVolume", "opening", "heatTransfer.Ts", "medium.t", "port_a.p", "port_b.p", "condition", "open", "N_in",]

for column_name in df1.columns:
    column_values = df1[column_name][0]
    print(f"Column Name: {column_name}")
    print("Column Values:")
    print(column_values)
    print()
    


#for df in df_list:
#    plot_fig(df)
  
