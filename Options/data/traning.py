# check correlations of squared observations
import csv
import math
from random import gauss
from random import seed
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np

returns = []
with open("cleaned_data.csv") as f:
    data = csv.reader(f)
    previous = 0.0
    prev_time = ''
    j = 0
    i = 0
    prevj = 0
    for  row in data:
        #print(i,j)

        if i < 6:
            i += 1
            continue
        if(row[1] != prev_time):
            prevj = j
            j +=1
            current =  (float(row[15])+float(row[16]))/2
            if i == 6:
                previous = current
                prev_time = row[1]
                continue
            returns.append(abs(math.log(current/previous)))
            previous = current
            prev_time = row[1]
        i += 1
        if (j == 10081):
            break
returns[0] = 0.0
print(len(returns))
returns = np.array(returns)
print(len(returns))
plot_acf(returns)
pyplot.savefig("training2.pdf")
pyplot.show()