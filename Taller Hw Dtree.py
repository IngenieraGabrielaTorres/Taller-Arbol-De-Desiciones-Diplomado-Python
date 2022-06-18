#Taller 
#Gariela Torres
#Decision Tree
#ID:1001970935
#ID:502193
#correo:gabriela.torresr@upb.edu.co
#Cel:3234708201
#Diplomado de PYTHON APLICADO A LA INGENIERIA 
#Docente:Roberto Paez Salgado
#Modulo 2

#importamos las librerias

import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import pydotplus

#Creamos el dataframe
carseats = sm.datasets.get_rdataset("Carseats", "ISLR")
mis_datos = carseats.data
print(carseats.__doc__)

mis_datos['ventas_altas'] = np.where(mis_datos.Sales > 8, 0, 1)
mis_datos = mis_datos.drop(columns = 'Sales')

dsheveloc= {'Bad':0, 'Medium':1, 'Good':2}
durban= {'Yes':1, 'No':0}
dus= {'Yes':1, 'No':0}

mis_datos["ShelveLoc"] = mis_datos["ShelveLoc"].map(dsheveloc)
mis_datos["Urban"] = mis_datos["Urban"].map(durban)
mis_datos["US"] = mis_datos["US"].map(dus)


features = ["CompPrice","Income","Advertising", "Population","Price","ShelveLoc","Age","Education","Urban","US"]

X = mis_datos[features]
y = mis_datos["ventas_altas"]

x_train = X[:320]
y_train = y[:320]

x_test = X[320:]
y_test = y[320:]

dtree = DecisionTreeClassifier()
dtree  = dtree.fit(x_train, y_train)

prediccion = dtree.predict([[136,70,12,171,152,1,44,18,1,1]])

dato = tree.export_graphviz(dtree, out_file = None, feature_names= features)
graph = pydotplus.graph_from_dot_data(dato)
graph.write_png('mydecisiontree.png')
img = pltimg.imread("mydecisiontree.png")
imgplot = plt.imshow(img)
plt.show()