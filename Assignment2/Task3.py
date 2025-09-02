import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

INCH_CM = 2.54
POUND_KG = 0.453592

df = pd.read_csv('../AI with python/weight-height.csv')
data = df.to_numpy()
print(df,data)
length = data[:,1]
weight = data[:,2]
inches_to_cm = length * INCH_CM
pounds_to_kgs = weight * POUND_KG
print("Length in inches\n",length,"\nLength in cm\n",inches_to_cm,"\nweigth in pounds\n",weight,"\nweight in kgs\n",pounds_to_kgs)
mean_length = np.mean(inches_to_cm)
mean_weight = np.mean(pounds_to_kgs)
print("Mean length = ", mean_length, "Mean weight = ", mean_weight)
plt.hist(length,bins=30, color="red", edgecolor="black")
plt.xlabel("Heights(inches)")
plt.ylabel("Number of students")
plt.title("Heights of students")
plt.show()



