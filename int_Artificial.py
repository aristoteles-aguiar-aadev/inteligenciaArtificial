
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Usando o Pandas para importar o arquivo .csv
tabela = pd.read_csv(r"D:\Aula de Python - Hashtag\Aula 4\advertising.csv")
print(tabela)

# outra forma de ver a mesma análise
# sns.pairplot(tabela)
# plt.show()

sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")
plt.show()

# Separando em dados de treino e dados de teste
y = tabela["Vendas"]
x = tabela.drop("Vendas", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)


# cria as inteligencias artificiais
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treina as inteligencias artificiais
modelo_regressaolinear.fit(X_train, y_train)
modelo_arvoredecisao.fit(X_train, y_train)

# Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece

# criar metricas de previsoes
previsao_regressaolinear = modelo_regressaolinear.predict(X_test)
previsaoarvoredecisao = modelo_arvoredecisao.predict(X_test)

# comparar os modelos
print(metrics.r2_score(y_test, previsao_regressaolinear))
print(metrics.r2_score(y_test, previsaoarvoredecisao))
 
# visualizacao grafica das previsoes
tabela_auxilar = pd.DataFrame()
tabela_auxilar["y,test"] = y_test
tabela_auxilar["Previsao Arvore Decisao"] = previsaoarvoredecisao
tabela_auxilar["Previsao Regressao Linear"] = previsao_regressaolinear

plt.figure(figsize=(20, 8))
sns.lineplot(data=tabela_auxilar)
plt.show()

# Como fazer uma nova previsao
# importar a nova_tabela com o pandas (a nova tabela tem que ter os dados de TV, Radio e Jornal)
# previsao = modelo_randomforest.predict(nova_tabela)
# print(previsao)

nova_tabela = pd.read_csv(r"D:\Aula de Python - Hashtag\Aula 4\novos.csv")
print(nova_tabela)
previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)

sns.barplot(x=X_train.columns, y=modelo_arvoredecisao.feature_importances_)
plt.show()

# Caso queira comparar Radio com Jornal
# print(df[["Radio", "Jornal"]].sum())