#import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import scipy.optimize as opt
import err_ranges as err


def read_file(file_name):
    """This function reads the data"""
    df = pd.read_csv(file_name)
    """Dropping the columns"""
    dd = df.drop(columns=["Series Name","Country Name","Country Code"])
    """Change the null values to 0"""
    dd = dd.replace(np.nan,0)
    dt = np.transpose(dd)
    dt = dt.reset_index()
    dt = dt.rename(columns={"index":"year", 0:"UNITED STATES", 1:"INDIA"})
    dt= dt.iloc[1:]
    dt = dt.dropna()
    
    
    dt["year"] = dt["year"].str[:4]
    dt["year"] = pd.to_numeric(dt["year"])
    dt["INDIA"] = pd.to_numeric(dt["INDIA"])
    dt["UNITED STATES"] = pd.to_numeric(dt["UNITED STATES"])
    print(dt)
    return df, dt

#clustering of USA and India -Agriculture
df_agr,df_agrt= read_file("C:/Users/hp/Desktop/last/AGRICULTURE  IN.csv")
df_for,df_forst = read_file("C:/Users/hp/Desktop/last/FOREST IN.csv")
df_agrt= df_agrt.iloc[:,1:3]

kmean = cluster.KMeans(n_clusters=2).fit(df_agrt)
label = kmean.labels_
plt.scatter(df_agrt["UNITED STATES"],df_agrt["INDIA"],c=label,cmap="jet")
plt.title("USA and India -Agriculture")
c = kmean.cluster_centers_
plt.savefig("Scatter.png", dpi = 300, bbox_inches='tight')
plt.show()
#clustering:for agriculture vs forest area -India
def clustering(df_agrt):
      india = pd.DataFrame()
      india["AGRICULTURE"] = df_agrt["INDIA"]
      india["FOREST"] =df_forst ["INDIA"]



      kmean = cluster.KMeans(n_clusters=2).fit(india)
      label = kmean.labels_
      plt.scatter(india["agriculture"],india["forest"],c=label,cmap="jet")
      plt.title("agriculture vs forest area -India")
c = kmean.cluster_centers_

for t in range(2):
    xc,yc = c[t,:]
    plt.plot(xc,yc,"dk",markersize=10)
plt.figure()
plt.savefig("Scatter for agriculture land vs forest area -India .png", dpi = 300, bbox_inches='tight')
plt.show()
"""DATA FITTING"""


def curve(t, scale, growth):
    f = scale * np.exp(growth * (t-1960))
    return f
df_renw, df_renwt = read_file("C:/Users/hp/Desktop/last/Renewable (2) IN.csv")

param, cov = opt.curve_fit(curve,df_renwt["year"],df_renwt["INDIA"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
low,up = err.err_ranges(df_renwt["year"],curve,param,sigma)

df_renwt["fit_value"] = curve(df_renwt["year"], * param)
plt.figure()
plt.title("Renewable energy use as a percentage of total energy - India")
plt.plot(df_renwt["year"],df_renwt["INDIA"],label="data")
plt.plot(df_renwt["year"],df_renwt["fit_value"],c="red",label="fit")
plt.fill_between(df_renwt["year"],low,up,alpha=0.5)
plt.legend()
plt.xlim(1990,2019)
plt.xlabel("Year")
plt.ylabel("Renewable energy(% of total energy use)")
plt.savefig("Renewable_India.png", dpi = 300, bbox_inches='tight')
plt.show()



param, cov = opt.curve_fit(curve,df_renwt["year"],df_renwt["UNITED STATES"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
low,up = err.err_ranges(df_renwt["year"],curve,param,sigma)

df_renwt["fit_value"] = curve(df_renwt["year"], * param)
plt.figure()
plt.title("Renewable energy prediction - UNITED STATES")
pred_year = np.arange(1980,2030)
pred_ind = curve(pred_year,*param)
plt.plot(df_renwt["year"],df_renwt["UNITED STATES"],label="data")
plt.plot(pred_year,pred_ind,label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Renewable energy(% of total energy use)")
plt.savefig("Renewable_Prediction_UNITED STATES.png", dpi = 300, bbox_inches='tight')
plt.show()


plt.savefig("India_Predicted.png", dpi = 300, bbox_inches='tight')
plt.show()


