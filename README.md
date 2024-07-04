---

# Optimización de Inventario con Redes Neuronales de la ferretería roberto cotlear
<img src="https://www.ferreteriarobertocotlear.tech/public/img/logo.png"/>

Este proyecto utiliza redes neuronales artificiales para la **optimización de inventario** en una ferretería. La solución busca predecir la demanda de productos utilizando datos históricos de ventas y configuraciones de modelos de aprendizaje automático.

## 📋 Descripción

Este repositorio contiene un código Python que entrena un modelo de red neuronal para predecir la cantidad demandada de productos en función de datos históricos de ventas. El modelo se entrena con diferentes configuraciones para mejorar la precisión de las predicciones, y se visualizan los resultados para seleccionar la mejor configuración.

## 🚀 Instalación

Para ejecutar el código en este repositorio, necesitarás tener Python 3.x y los siguientes paquetes instalados. Puedes instalar todos los paquetes necesarios usando `pip`:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

## 🔧 Uso

A continuación, te mostramos cómo usar el código proporcionado para entrenar el modelo de red neuronal y evaluar diferentes configuraciones.

1. **Clona el repositorio:**

    ```bash
    git clone https://github.com/ManuelNZ20/proyecto-python-sistemas-intelingenes.git
    cd proyecto-python-sistemas-intelingenes
    ```

2. **Ejecuta el script principal:**

    ```bash
    python intelligence_system_presentation.py
    python intelligence_system.py
    ```

    Este script realiza los siguientes pasos:
    - Prepara los datos históricos de ventas.
    - Convierte las características categóricas a representaciones numéricas.
    - Define y entrena el modelo de red neuronal con varias configuraciones.
    - Evalúa el rendimiento del modelo en el conjunto de prueba.
    - Muestra gráficos de las métricas del modelo para diferentes configuraciones.

## 📝 Código

El archivo principal del repositorio es `optimizacion_inventario.py` y contiene el siguiente flujo de trabajo:

### 1. Preparación de Datos

Se crea un DataFrame con datos históricos de ventas y se procesan para la entrada del modelo.

```python
data = [
    # Datos históricos de ventas
]
df = pd.DataFrame(data, columns=['order_id', 'order_code', 'product_id', 'quantity', 'price'])
```

### 2. Procesamiento de Datos

Se convierte `quantity` a numérico, se aplican codificaciones one-hot para las variables categóricas y se dividen los datos en conjuntos de entrenamiento y prueba.

```python
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
df_encoded = pd.get_dummies(df, columns=['product_id', 'order_code'])
X = df_encoded.drop(columns=['quantity'])
y = df_encoded['quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Definición y Entrenamiento del Modelo

Se definen varias configuraciones del modelo de red neuronal y se entrenan.

```python
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
```

### 4. Evaluación del Modelo

Se evalúan los modelos con diferentes configuraciones y se visualizan las métricas.

```python
loss, mae = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Mean Absolute Error: {mae}')
```

### 5. Visualización de Resultados

Se generan gráficos para comparar las métricas del modelo entre diferentes configuraciones.

```python
plt.figure(figsize=(14, 6))
# Gráfico de MAE
plt.subplot(1, 2, 1)
# Gráfico de Loss
plt.subplot(1, 2, 2)
plt.tight_layout()
plt.show()
```

## 📊 Gráficos

El script genera los siguientes gráficos:

- **MAE por Configuración del Modelo**: Muestra el Error Absoluto Medio para diferentes configuraciones.
- **Loss por Configuración del Modelo**: Muestra la Pérdida para diferentes configuraciones.

## 📝 Ejemplo de Salida

Aquí tienes ejemplos de salida del script:

```
Loss: 5423.54736328125, Mean Absolute Error: 51.117652893066406
Actual: 205.0, Predicted: 78.4365463256836
Actual: 2.0, Predicted: 6.331133842468262
...
```

## 📜 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Si tienes sugerencias o encuentras errores, abre un *issue* o envía un *pull request*.

## 📧 Contacto

Si tienes preguntas, puedes contactarme en [tu_email@example.com](mailto:tu_email@example.com).

---

**¡Gracias por tu interés en el proyecto!**

### **Archivo `optimizacion_inventario.py` Completo**

Aquí tienes el contenido completo del archivo `optimizacion_inventario.py` para referencia:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Paso 1: Preparar los datos
data = [
    # (order_id, order_code, product_id, quantity, price)
    # (datos omitidos por brevedad)
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

# Configuraciones del modelo
configurations = [
    {'layers': [64, 32], 'learning_rate': 0.001},
    {'layers': [128, 64], 'learning_rate': 0.001},
    {'layers': [64, 32, 16], 'learning_rate': 0.001},
    {'layers': [128, 64, 32], 'learning_rate': 0.001},
    {'layers': [64, 32], 'learning_rate': 0.01},
    {'layers': [128, 64], 'learning_rate': 0.01},
    {'layers': [64, 32, 16], 'learning_rate': 0.01},
    {'layers': [128, 64, 32], 'learning_rate': 0.01},
]

results = []

for config in configurations:
    model = Sequential()
    model.add(Dense(config['layers'][0], input_dim=X_train.shape[1], activation='relu'))
    for neurons in config['layers'][1:]:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=0)

    # Evaluar el modelo
    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    # Guardar resultados
    results.append({
        'layers': config['layers'],
        'learning_rate': config['learning_rate'],
        'loss': loss,
        'mae': mae,
    })

# Crear un DataFrame con los resultados
results_df = pd.DataFrame(results)

# Graficar resultados
plt.figure(figsize=(14, 6))

# Gráfico de MAE para diferentes configuraciones
plt.subplot(1, 2, 1)
for label, df_group in results_df.groupby(['learning_rate']):
    plt.plot(df_group['layers'].apply(lambda x: '-'.join(map(str, x))), df_group['mae'], marker='o', label=f'LR={label}')
plt.xticks(rotation=45)
plt.xlabel('Configuraciones de Capas')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE por Configuración del Modelo')
plt.legend()

# Gráfico de Loss para diferentes configuraciones
plt.subplot(1, 2, 2)
for label,

 df_group in results_df.groupby(['learning_rate']):
    plt.plot(df_group['layers'].apply(lambda x: '-'.join(map(str, x))), df_group['loss'], marker='o', label=f'LR={label}')
plt.xticks(rotation=45)
plt.xlabel('Configuraciones de Capas')
plt.ylabel('Loss')
plt.title('Loss por Configuración del Modelo')
plt.legend()

plt.tight_layout()
plt.show()
```
# 👏GRACIAS POR LEER EL README 😉
