# --- Missão 10.4: O Batismo de Fogo (Versão Final) ---
# Objetivo: Treinar e avaliar a nossa CNN de Elite com o dataset MNIST.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- FASE 1: LOGÍSTICA DE DADOS ---
print("--- FASE 1: Carregando e preparando a inteligência de campo... ---")
(imagens_treino, rotulos_treino), (imagens_teste, rotulos_teste) = tf.keras.datasets.mnist.load_data()

imagens_treino = imagens_treino / 255.0
imagens_teste = imagens_teste / 255.0

imagens_treino = np.expand_dims(imagens_treino, -1)
imagens_teste = np.expand_dims(imagens_teste, -1)

print("Dados carregados e pré-processados com sucesso.")

# --- FASE 2: CONSTRUÇÃO DA ARQUITETURA ---
print("\n--- FASE 2: Forjando a arquitetura da CNN... ---")
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Arquitetura forjada com sucesso.")
model.summary()

# --- FASE 3: TREINAMENTO (O BATISMO DE FOGO) ---
print("\n--- FASE 3: INICIANDO CAMPANHA DE TREINAMENTO ---")

model.fit(imagens_treino, rotulos_treino, epochs=5)

# --- FASE 4: AVALIAÇÃO DE DESEMPENHO ---
print("\n--- FASE 4: AVALIANDO O DESEMPENHO EM CAMPO DE BATALHA DESCONHECIDO ---")

loss, accuracy = model.evaluate(imagens_teste, rotulos_teste, verbose=2)

# --- RELATÓRIO FINAL DE MISSÃO (SINTAXE CORRIGIDA) ---
print("\n--- RELATÓRIO FINAL DE MISSÃO ---")
# A CORREÇÃO ESTÁ AQUI: Usamos ':.2f' em Python para formatar o número.
print(f"Precisão final do modelo no território de teste: {accuracy * 100:.2f}%")