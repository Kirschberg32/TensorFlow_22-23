import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

name = "all"

list_of_data_path = ["original"]

for p in list_of_data_path:
    path = f"Plots/{p}.png"

    f = pd.read_csv(path)
    print(f)

#plt.figure(figsize=(16, 6))
#sns.lineplot(data=data_frame,x="step",y="value",hue="tag",style="mode").set_title("results")
#plt.savefig(f"Plots/{name}.png")
#plt.show()