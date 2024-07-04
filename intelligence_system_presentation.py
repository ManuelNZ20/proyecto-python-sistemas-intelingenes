import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Paso 1: Preparar los datos
data = [
    (9, 'f6391150-8', 'C605BB1F', 2, 80),
    (11, 'f6391150-8', 'C605BB1F', 2, 80),
    (15, '1388eba0-8', 'C605BB1F', 2, 80),
    (17, '526bb1e9-8', 'C605BB1F', 2, 80),
    (19, 'a01607f8-8', 'C605BB1F', 2, 80),
    (21, 'd04cc1e2-8', 'C605BB1F', 2, 80),
    (22, '195474a4-8', '52B6FD6A', 0, 44),
    (23, '195474a4-8', '4938E7FB', 0, 30),
    (26, 'd36635a3-8', '52B6FD6A', 0, 44),
    (27, 'd36635a3-8', '4938E7FB', 0, 30),
    (29, 'aee5ac55-8', 'C762B287', 23, 0),
    (31, '07bfd0d2-8', 'C605BB1F', 80, 2),
    (32, '8f4bb061-8', '9EDEF19E', 24.7, 2),
    (34, 'a590ea4f-8', 'C762B287', 23, 1),
    (35, '7c176434-8', '52B6FD6A', 44, 2),
    (36, 'b2c3122e-8', '52B6FD6A', 44, 1),
    (37, 'e2db82f1-8', '52B6FD6A', 44, 1),
    (38, 'e2b22c2f-8', '4938E7FB', 30, 1),
    (39, 'e2b22c2f-8', '017EDD5C', 27, 3),
    (43, '21c0c56d-8', '52B6FD6A', 44, 1),
    (50, '57973aea-8', 'E614F5A6', 60, 3),
    (51, '57973aea-8', '52B6FD6A', 44, 3),
    (53, '57973aea-8', 'AF470DA5', 32.5, 2),
    (55, '04a76aa1-8', '017EDD5C', 27, 1),
    (56, 'e746c85f-8', '52B6FD6A', 44, 3),
    (57, 'e746c85f-8', '80379AD0', 38.5, 3),
    (58, '3cb0f041-8', '52B6FD6A', 44, 3),
    (59, '3cb0f041-8', '80379AD0', 38.5, 1),
    (60, 'd0bd8526-8', 'C605BB1F', 80, 1),
    (61, 'd0bd8526-8', '52B6FD6A', 44, 2),
    (62, 'f5d15a68-8', 'C605BB1F', 80, 1),
    (63, 'f5d15a68-8', '52B6FD6A', 44, 2),
    (64, '04bc45b0-8', 'C605BB1F', 80, 1),
    (65, '04bc45b0-8', '52B6FD6A', 44, 2),
    (66, '4cd30634-8', 'AF470DA5', 32.5, 1),
    (67, '64a29d31-8', 'AF470DA5', 32.5, 1),
    (68, '39dd8a70-8', 'AF470DA5', 32.5, 2),
    (70, '6f3b059c-8', 'AF470DA5', 32.5, 4),
    (71, '4e5978f8-8', 'AF470DA5', 32.5, 4),
    (72, '5a4f2547-8', 'AF470DA5', 32.5, 6),
    (75, '94863444-9', 'B3D127AC', 205, 1),
    (78, '66c5c65f-9', 'B3D127AC', 205, 1),
    (79, '8c0acf3f-9', 'B3D127AC', 205, 1),
    (80, '0036378e-9', 'B3D127AC', 205, 1),
    (81, '1790308f-9', 'B3D127AC', 205, 1),
    (82, '18575cc1-9', 'B3D127AC', 205, 1),
    (83, '1b646904-9', 'B3D127AC', 205, 1),
    (84, '249f65ed-9', 'C605BB1F', 80, 1),
    (106, 'ad6cfa96-9', '52B6FD6A', 44, 1),
    (109, 'd906ac69-9', '52B6FD6A', 44, 1),
    (110, 'd9d64fe7-9', '52B6FD6A', 44, 1),
    (111, 'fca76d5f-9', '9EDEF19E', 24.7, 1),
    (112, '06057310-9', '9EDEF19E', 24.7, 1),
    (113, '06873fe2-9', '9EDEF19E', 24.7, 1),
    (114, '2c9c76f3-9', '9EDEF19E', 24.7, 1),
    (119, '3cbaee1d-9', '9EDEF19E', 24.7, 1),
    (120, '8b853559-9', '9EDEF19E', 24.7, 1),
    (121, 'a4b5d575-9', '9EDEF19E', 24.7, 1),
    (122, 'ab385a90-9', '9EDEF19E', 24.7, 1),
    (123, 'ab89964d-9', '9EDEF19E', 24.7, 1),
    (124, 'abe155b7-9', '9EDEF19E', 24.7, 1),
    (128, '7af2f25d-9', 'E614F5A6', 60, 1),
    (129, 'd10eef98-9', '52B6FD6A', 44, 1),
    (130, 'fa795ebb-9', '52B6FD6A', 44, 1),
    (131, '03edba76-9', '52B6FD6A', 44, 1),
    (132, '67751e86-9', 'C605BB1F', 80, 1),
    (133, '5ba77e03-9', 'B3D127AC', 205, 1),
    (135, '6a31081e-8', '017EDD5C', 27, 1),
    (137, '4d2e8c7e-8', '80379AD0', 38.5, 1),
    (138, '8ceebd9f-8', '52B6FD6A', 44, 1),
    (139, '8ceebd9f-8', '4938E7FB', 30, 1),
    (140, 'cf68315a-8', 'C605BB1F', 80, 2),
    (141, 'cf68315a-8', 'E614F5A6', 60, 2),
    (142, 'cf68315a-8', 'D878FFE0', 59.99, 2),
    (143, '25834776-8', 'C605BB1F', 80, 2),
    (144, '25834776-8', 'E614F5A6', 60, 2),
    (145, '25834776-8', 'D878FFE0', 59.99, 2),
    (146, '6a5e0557-8', '75D59241', 29.9, 1),
    (147, '6a5e0557-8', '755252FD', 49, 1),
    (148, '8ef8e126-8', '688A8FEE', 44, 3),
    (149, '8ef8e126-8', '75D59241', 29.9, 1),
    (151, '2260c025-8', 'B3D127AC', 205, 1),
    (152, '2260c025-8', '755252FD', 49, 1),
    (153, '2260c025-8', '688A8FEE', 44, 1),
    (154, '6a3e7840-8', 'B3D127AC', 205, 1),
    (155, '6a3e7840-8', '755252FD', 49, 1),
    (156, '6a3e7840-8', '688A8FEE', 44, 1),
    (158, 'c65e55ea-8', 'D878FFE0', 59.99, 88),
    (159, 'e120c748-8', 'D878FFE0', 59.99, 88),
    (160, 'e94fa36c-8', 'D878FFE0', 59.99, 80),
    (161, '10efc98f-8', '52B6FD6A', 44, 3),
    (162, 'bc8ee643-8', 'C762B287', 23, 2),
    (163, '781c8828', '755252FD', 49, 1),
    (164, '19826b92', '755252FD', 49, 1),
    (165, '01df84f0', '755252FD', 49, 1),
    (166, 'aa140e1d', '755252FD', 49, 1),
    (167, '3a8ced50', '755252FD', 49, 1),
    (168, 'f045e52e', '755252FD', 49, 1),
    (169, 'f8caf583', '75D59241', 29.9, 1),
    (170, 'f8b1bd3e', '755252FD', 49, 1),
    (171, '50640428', '755252FD', 49, 1),
    (172, '6e90bc3e', '755252FD', 49, 1),
    (173, '9fb47d70', '755252FD', 49, 1),
    (174, 'a81e25a5', '755252FD', 49, 1),
    (175, 'bef5344c', '755252FD', 49, 1),
    (176, '1612684c', '755252FD', 49, 1),
    (177, 'd6d7ac8d', '755252FD', 49, 1),
    (180, '4e75cc23', 'CB686ECB', 219, 1),
    (181, '4e75cc23', '94AEDA30', 5, 1),
    (182, '4e75cc23', '2C197048', 34, 1),
    (184, 'b37d81d4', '39B922C5', 229, 1),
    (186, '9a776a88-9', 'B3D127AC', 205, 1),
    (191, 'be9090ed-9', '688A8FEE', 44, 1),
    (192, '8901b414-9', 'CB686ECB', 219, 1),
    (194, '7709e31c-9', '688A8FEE', 44, 1),
]

# Convertir los datos a un DataFrame de pandas
df = pd.DataFrame(data, columns=['order_id', 'order_code', 'product_id', 'quantity', 'price'])

# Agrupar por 'product_id' y sumar las cantidades
product_demand = df.groupby('product_id')['quantity'].sum().reset_index()

# Ordenar por cantidad en orden descendente
product_demand = product_demand.sort_values(by='quantity', ascending=False).reset_index(drop=True)



# Asegurarnos de que 'quantity' es numérico
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

# Convertir las columnas categóricas a numéricas usando one-hot encoding
df_encoded = pd.get_dummies(df, columns=['product_id', 'order_code'])

# Definir las características (X) y las etiquetas (y)
X = df_encoded.drop(columns=['quantity'])
y = df_encoded['quantity']

# Asegurarnos de que X e y son matrices numpy
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir la arquitectura de la red neuronal
model = Sequential()
# PRIMERA CAPA
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
# SEGUNDA CAPA
model.add(Dense(32, activation='relu'))
# TERCERA CAPA
model.add(Dense(1, activation='linear'))

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)


# Evaluar el modelo
loss, mae = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Mean Absolute Error: {mae}')

# Predecir con el modelo
y_pred = model.predict(X_test)

# Mostrar algunas predicciones
for i in range(10):
    print(f'Actual: {y_test[i]}, Predicted: {y_pred[i][0]}')



# Mostrar el resultado
product_demand.head(10)