import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

#list of config dictionaries you want to get the data from
list_of_data_path = ["Original","all", "Aug", "Dropout_0.5", "Aug_Dropout0.5", "Normalisation", "Reg_L2", "ResMode", "DenseMode"]
#print the graphs as images
print(130*"_")
for p in list_of_data_path:
    path = f"Plots/{p}/"
    files = os.listdir(f"csvs/{p}/")[0] #first file in path. Should be last but there are invisible .ipynb checkpointfiles which are problematic 
    file_path = path + files + ".png"
    #f = os.read(file_path)
    print(file_path)
    Image.open(file_path).show()
    print(130*"_")

#plt.figure(figsize=(16, 6))
#sns.lineplot(data=data_frame,x="step",y="value",hue="tag",style="mode").set_title("results")
#plt.show()