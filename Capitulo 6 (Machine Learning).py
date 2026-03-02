# =========================================================
# FUNDAMENTOS DE MACHINE LEARNING - IMPLEMENTAÇÃO DIDÁTICA
# =========================================================

# Biblioteca matemática:
# - sqrt(): usado no RMSE
# - log(): usado no Log Loss
import math


# =========================================================
# ---------------------- REGRESSÃO --------------------------
# =========================================================
# Regressão é usada quando queremos prever valores numéricos contínuos.
# Exemplo: preço de casa, temperatura, lucro, demanda, etc.
# Saída do modelo: um número (ex.: 1520.75)

print("\n--- Regressão ---")


# ================= MODELO LINEAR =================
# Modelo linear é a forma mais básica de regressão.
# Fórmula geral:
#   y_hat = w1*x1 + w2*x2 + ... + b
# Onde:
#   w = pesos (o quanto cada variável influencia a previsão)
#   b = bias/intercepto (ajuste "base" quando x = 0)

w1, w2, w3, b = 0.5, 1.2, -0.3, 2
x1, x2, x3 = 10, 5, 8

# Cálculo da previsão (soma ponderada + bias)
y_pred = w1 * x1 + w2 * x2 + w3 * x3 + b

print("\nPrevisão (Modelo de Regressão Linear) =", y_pred)


# ================= MSE =================
# MSE = Mean Squared Error (Erro Quadrático Médio)
# Mede o erro médio ao quadrado entre valor real e previsto.
# - Penaliza erros grandes (porque eleva ao quadrado)
# - Muito usado em regressão e em redes neurais
#
# Risco Empírico:
# - "Risco empírico" é a MÉDIA da loss no conjunto de dados.
# - Aqui, como a loss é erro quadrático, o risco empírico = MSE.

y = [10, 5, 8]     # valores reais (ground truth)
y_hat = [8, 6, 9]  # previsões do modelo
n = len(y)         # número de observações


def mse_loss(y_real, y_pred):
    # Loss por amostra: (erro)^2
    return (y_real - y_pred) ** 2


# Média da loss nas n amostras => risco empírico
mse = sum(mse_loss(y[i], y_hat[i]) for i in range(n)) / n

print("\nRisco Empírico (MSE) =", mse)


# ================= R² =================
# R² = Coeficiente de Determinação
# Interpretação:
# - Mede quanto da variação natural de y é explicada pelo modelo
# - R² = 1  -> perfeito
# - R² = 0  -> tão bom quanto prever sempre a média de y
# - R² < 0  -> pior que prever sempre a média
#
# Fórmula:
#   R² = 1 - (SS_res / SS_tot)
# Onde:
#   SS_res = soma dos erros ao quadrado (resíduos)
#   SS_tot = soma da variação de y em torno da média

y = [3, 5, 7, 9]
y_hat = [2.5, 5.5, 6.5, 10]
n = len(y)

y_mean = sum(y) / n  # média de y

# Erro do modelo (quanto o modelo erra)
ss_res = sum((y[i] - y_hat[i]) ** 2 for i in range(n))

# Variação total dos dados (o "quanto y varia naturalmente")
ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))

r2 = 1 - (ss_res / ss_tot)

print("\nCoeficiente de Determinação (R²) =", r2)


# ================= MAE =================
# MAE = Mean Absolute Error (Erro Absoluto Médio)
# - Usa valor absoluto do erro
# - Menos sensível a outliers do que o MSE
# - Interpretação bem intuitiva: "erro médio" na unidade do dado

def mae_loss(y_real, y_pred):
    return abs(y_real - y_pred)


mae = sum(mae_loss(y[i], y_hat[i]) for i in range(n)) / n

print("\nRisco Empírico (MAE) =", mae)


# ================= HUBER LOSS =================
# Huber Loss = mistura de MSE e MAE
# Ideia:
# - Para erros pequenos: comporta como MSE (suave/diferenciável)
# - Para erros grandes: comporta como MAE (robusta a outliers)
#
# delta controla o ponto de transição:
# - delta pequeno -> mais parecido com MAE
# - delta grande  -> mais parecido com MSE

delta = 1


def huber_loss(y_real, y_pred):
    error = y_real - y_pred  # erro (resíduo)

    # Região quadrática (tipo MSE)
    if abs(error) <= delta:
        return 0.5 * error ** 2
    # Região linear (tipo MAE)
    else:
        return delta * (abs(error) - 0.5 * delta)


# Média da Huber loss (risco empírico para Huber)
huber = sum(huber_loss(y[i], y_hat[i]) for i in range(n)) / n

print("\nRisco Empírico (Huber) =", huber)


# ================= RMSE =================
# RMSE = Root Mean Squared Error
# - É a raiz do MSE
# - Vantagem: volta para a mesma unidade do dado original
#   (ex.: se y está em reais, RMSE também estará em reais)
# - Penaliza erros grandes (porque vem do MSE)

rmse = math.sqrt(mse)

print("\nRMSE =", rmse)


# =========================================================
# -------------------- CLASSIFICAÇÃO ------------------------
# =========================================================
# Classificação é usada quando queremos prever categorias.
# Exemplo: spam/não spam, fraude/não fraude, 0/1, etc.
# Saída do modelo: classe (ex.: 0 ou 1) ou probabilidade.

print("\n--- Classificação ---")


# ================= 0-1 LOSS =================
# 0-1 Loss por amostra:
# - 0 se acertou
# - 1 se errou
#
# Quando tiramos a MÉDIA no dataset, obtemos a taxa de erro (error rate):
# - valor entre 0 e 1
# - ex.: 0.25 significa 25% das previsões erraram

y = [1, 0, 1, 1]
y_hat = [1, 1, 1, 0]
n = len(y)


def zero_one_loss(y_real, y_pred):
    return 1 if y_real != y_pred else 0


error_rate = sum(zero_one_loss(y[i], y_hat[i]) for i in range(n)) / n

print("\nRisco Empírico (0-1 Loss) =", error_rate)


# ================= LOG LOSS =================
# Log Loss (Cross-Entropy) é usada quando o modelo prevê PROBABILIDADES.
# Ela mede o quão boas são as probabilidades previstas.
#
# Ideia:
# - Se o modelo estiver muito confiante e errado -> penalização muito alta
# - Se estiver confiante e certo -> penalização baixa
#
# Observação importante:
# - prob deve estar entre 0 e 1 (e não pode ser 0 ou 1 exatos, por causa do log)

y = [1, 0, 1]
p = [0.9, 0.2, 0.8]
n = len(y)


def log_loss(y_real, prob):
    return -(y_real * math.log(prob) + (1 - y_real) * math.log(1 - prob))


logloss = sum(log_loss(y[i], p[i]) for i in range(n)) / n

print("\nFunção de Perda (Log Loss) =", logloss)


# =========================================================
# ------------- AVALIAÇÃO DE CLASSIFICAÇÃO -----------------
# =========================================================
# Métricas clássicas para avaliar classificadores.
# Importante: elas são baseadas na matriz de confusão:
# TP, TN, FP, FN

print("\n--- Avaliação de Classificação ---")

y = [1, 0, 1, 1, 0]
y_hat = [1, 1, 1, 0, 0]
n = len(y)

# Accuracy:
# proporção total de acertos
accuracy = sum(1 for i in range(n) if y[i] == y_hat[i]) / n
print("\nAccuracy =", accuracy)

# Matriz de confusão:
# TP: previu 1 e era 1
tp = sum(1 for i in range(n) if y[i] == 1 and y_hat[i] == 1)

# TN: previu 0 e era 0
tn = sum(1 for i in range(n) if y[i] == 0 and y_hat[i] == 0)

# FP: previu 1, mas era 0 (falso alarme)
fp = sum(1 for i in range(n) if y[i] == 0 and y_hat[i] == 1)

# FN: previu 0, mas era 1 (perdeu um positivo)
fn = sum(1 for i in range(n) if y[i] == 1 and y_hat[i] == 0)

# Precision:
# dos "positivos previstos", quantos eram realmente positivos?
precision = tp / (tp + fp)

# Recall (sensibilidade):
# dos "positivos reais", quantos o modelo acertou?
recall = tp / (tp + fn)

# F1-score:
# média harmônica entre precision e recall (equilíbrio entre as duas)
f1 = 2 * (precision * recall) / (precision + recall)

print("Precision =", precision)
print("Recall =", recall)
print("F1-Score =", f1)

print("\nMatriz de Confusão")
print("TP =", tp)
print("TN =", tn)
print("FP =", fp)
print("FN =", fn)


# =========================================================
# ---------------- OTIMIZAÇÃO ------------------------------
# =========================================================
# Otimização é o processo de ajustar os pesos do modelo para reduzir a loss.
# Gradient Descent (Descida do Gradiente) atualiza w com:
#   w = w - eta * (dL/dw)
#
# - gradiente (dL/dw) diz a direção em que a loss aumenta
# - subtrair o gradiente faz ir na direção que diminui a loss
# - eta é o tamanho do passo (learning rate)

print("\n--- Otimização / Treinamento ---")

x = 2      # entrada
y = 10     # valor real
w = 1      # peso inicial
eta = 0.01 # taxa de aprendizado

# Previsão atual do modelo
y_hat = w * x

# Gradiente da MSE em relação a w (derivada)
# Para L = (y - wx)^2, temos dL/dw = -2*x*(y - y_hat)
grad = -2 * x * (y - y_hat)

# Atualização do peso (um passo do Gradient Descent)
w = w - eta * grad

print("\nPeso do modelo após otimização (Gradient Descent) =", w)


# =========================================================
# -------------------- REGULARIZAÇÃO -----------------------
# =========================================================
# Regularização adiciona uma "penalidade" aos pesos para reduzir overfitting.
# Intuição:
# - modelos com pesos muito grandes podem "memorizar" os dados
# - regularização incentiva pesos menores (modelo mais simples)
#
# L1 (Lasso): soma dos valores absolutos |w|
# - pode zerar pesos -> ajuda seleção de variáveis
#
# L2 (Ridge): soma dos quadrados w^2
# - reduz pesos grandes de forma suave

print("\n--- Regularização ---")

weights = [0.5, -1.2, 0.3]  # exemplo de pesos
lambda_ = 0.1               # força da penalização

# Penalização L1 (Lasso)
l1 = lambda_ * sum(abs(w) for w in weights)

# Penalização L2 (Ridge)
l2 = lambda_ * sum(w ** 2 for w in weights)

print("\nPenalização L1 (Lasso) =", l1)
print("Penalização L2 (Ridge) =", l2)


# =========================================================
# ------------------- GENERALIZAÇÃO ------------------------
# =========================================================
# Generalização = capacidade do modelo funcionar bem em dados novos (não vistos).
#
# Comparar erro de treino vs erro de teste:
# - erro_treino alto e erro_teste alto  -> underfitting (modelo fraco)
# - erro_treino baixo e erro_teste alto -> overfitting (decorou o treino)
# - erro_treino baixo e erro_teste baixo-> bom equilíbrio

print("\n--- Generalização do Modelo ---")

erro_treino = 0.05
erro_teste = 0.25

print("\nErro de Treino =", erro_treino)
print("Erro de Teste =", erro_teste)
