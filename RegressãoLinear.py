import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dados de exemplo
data = {
    'Tamanho': [1500, 1600, 1700, 1800, 1900],
    'Preço': [300000, 320000, 340000, 360000, 380000]
}
df = pd.DataFrame(data)

# Variáveis independentes e dependente
X = df[['Tamanho']]
y = df['Preço']

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Erro quadrático médio: {mse}')

# Coeficientes do modelo
print(f'Coeficiente: {model.coef_[0]}')
print(f'Intercepto: {model.intercept_}')
