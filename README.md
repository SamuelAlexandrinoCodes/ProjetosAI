# Projetos de Inteligência Artificial e Machine Learning

Bem-vindo ao meu repositório de projetos de IA! Este espaço é dedicado aos meus estudos e implementações práticas no fascinante mundo da Inteligência Artificial e do Machine Learning. Cada projeto aqui representa um passo na minha jornada de aprendizado, aplicando conceitos teóricos a problemas do mundo real.

## Projetos:

* **Regressão Linear com Descida do Gradiente (C#)**
  * **Descrição:** Uma implementação do zero do algoritmo de **Descida do Gradiente** para treinar um modelo de Regressão Linear Simples. O objetivo deste projeto é prever a nota de um aluno com base nas horas de estudo. O código demonstra o processo iterativo de ajuste dos parâmetros do modelo (inclinação `m` e intercepto `b`) para minimizar o erro (MSE - Mean Squared Error) e encontrar a linha que melhor se ajusta aos dados. É um estudo fundamental para entender o "coração" do treinamento de muitos modelos de Machine Learning.
  * **Como Executar:**
    1. Clone este repositório.
    2. Abra a solução do projeto no Visual Studio.
    3. Compile e execute. O console mostrará o progresso do treinamento a cada 100 épocas e, ao final, exibirá a equação da reta encontrada e uma previsão de exemplo.

* **Classificador de Spam com Vetores e Produto Escalar (C#)**
  * **Descrição:** Uma aplicação de console que implementa um classificador de spam simples. O objetivo é demonstrar a lógica matemática por trás de algoritmos de classificação de texto, utilizando conceitos como **vetorização de texto** e o **produto escalar** para medir a "semelhança" entre uma nova mensagem e um perfil de spam.
  * **Como Executar:**
    1. Abra a solução `ClassificadorDeSpam.sln` no Visual Studio.
    2. Compile e execute o projeto. O programa irá solicitar que insira uma mensagem e irá classificá-la como "Spam" ou "Não Spam".
    
* **Otimizador Multivariado com Descida do Gradiente (C#)**
  * **Descrição:** Uma evolução da Regressão Linear Simples, este projeto implementa um modelo que prevê a nota final de um aluno com base em múltiplas variáveis de entrada: horas de estudo, número de faltas e a nota de uma avaliação anterior. O núcleo do projeto é a implementação do algoritmo de **Descida do Gradiente Multivariado** para otimizar os múltiplos pesos (parâmetros) do modelo. Também demonstra uma etapa crucial de pré-processamento de dados: a **Normalização de Features (Z-score)**, essencial para que o algoritmo convirja de forma eficiente quando as variáveis têm escalas diferentes.
  * **Como Executar:**
    1. Clone este repositório.
    2. Abra a solução do projeto no Visual Studio.
    3. Compile e execute. O console exibirá as fases de normalização, os parâmetros finais encontrados após o treinamento e uma previsão para um novo aluno hipotético.

* *(Próximo projeto de IA)*