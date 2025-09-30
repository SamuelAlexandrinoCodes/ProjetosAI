# Projetos de Inteligência Artificial e Machine Learning

![C#](https://img.shields.io/badge/C%23-239120?style=for-the-badge&logo=c-sharp&logoColor=white)
![.NET](https://img.shields.io/badge/.NET-512BD4?style=for-the-badge&logo=.net&logoColor=white)
![Visual Studio](https://img.shields.io/badge/Visual%20Studio-5C2D91.svg?style=for-the-badge&logo=visual-studio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/SamuelAlexandrinoCodes/ProjetosAI?style=for-the-badge)

Este repositório serve como meu diário de bordo e portfólio na jornada de estudos em Inteligência Artificial e Machine Learning, com foco na construção de algoritmos a partir do zero para um entendimento profundo dos seus mecanismos internos.

---

## 📜 Índice

* [Rede Neural Multi-Camada (MLP) 'do Zero'](#rede-neural-multi-camada-mlp-do-zero-para-risco-de-diabetes-c)
* [Fábrica de Mini-Lotes (Mini-Batch GD)](#fábrica-de-mini-lotes-mini-batch-gd-para-classificação-de-íris-c)
* [Otimizador em Lote (Batch GD)](#otimizador-em-lote-batch-gd-para-diagnóstico-médico-c)
* [Classificador Logístico Binário 'do Zero'](#classificador-logístico-binário-do-zero-c)
* [E outros projetos fundamentais...](#projetos-fundamentais)

---

## 🚀 Projetos em Destaque

### Rede Neural Multi-Camada (MLP) 'do Zero' para Risco de Diabetes (C#)

* **Descrição:** A construção de uma Rede Neural Profunda (MLP) a partir do zero para resolver um problema de classificação de diagnóstico médico. Este projeto representa a transição para o Deep Learning, implementando uma Camada Oculta (ReLU), uma Camada de Saída (Sigmoide) e o algoritmo de **Backpropagation** para o treino.
* **Como Executar:**
    1.  Clone este repositório.
    2.  Abra a solução do projeto `RedeNeuralMLP` no Visual Studio.
    3.  Compile e execute. O console exibirá as fases do treino e o veredito para pacientes hipotéticos.

*(Aqui seria um local perfeito para um [GIF de demonstração] do console em ação)*

### Fábrica de Mini-Lotes (Mini-Batch GD) para Classificação de Íris (C#)
* **Descrição:** A implementação da tática de otimização padrão da indústria, a **Descida do Gradiente em Mini-Lote**. O código demonstra as manobras essenciais de **embaralhar (shuffle)** o dataset e dividi-lo em mini-lotes para um treino eficiente e estável no clássico dataset "Iris".
* **Como Executar:**
    1.  Clone este repositório.
    2.  Abra a solução do projeto `FabricaMiniLotes` no Visual Studio.
    3.  Compile e execute.

### Otimizador em Lote (Batch GD) para Diagnóstico Médico (C#)
* **Descrição:** Uma implementação da **Descida do Gradiente em Lote (Batch Gradient Descent)**, que processa todo o conjunto de dados para calcular um "gradiente médio" antes de cada atualização, resultando numa convergência mais suave e previsível.
* **Como Executar:**
    1.  Clone este repositório.
    2.  Abra a solução do projeto `OtimizadorEmLote` no Visual Studio.
    3.  Compile e execute.

---

## 🏛️ Projetos Fundamentais

<details>
<summary><strong>Clique para expandir e ver outros projetos de base</strong></summary>

#### Classificador Logístico Binário 'do Zero' (C#)
* **Descrição:** Uma implementação fundamental do algoritmo de **Regressão Logística** para classificação binária, construído inteiramente do zero, implementando a **Função Sigmóide** e a otimização via **Descida do Gradiente** para minimizar o **Log Loss**.
* **Como Executar:** Siga os passos padrões de clonar, abrir no Visual Studio e executar.

#### Regressão Linear com Descida do Gradiente (C#)
* **Descrição:** Uma implementação do zero do algoritmo de **Descida do Gradiente** para treinar um modelo de Regressão Linear Simples.
* **Como Executar:** Siga os passos padrões.

#### Classificador de Spam com Vetores e Produto Escalar (C#)
* **Descrição:** Uma aplicação que demonstra a lógica de classificação de texto utilizando **vetorização** e o **produto escalar**.
* **Como Executar:** Siga os passos padrões.

#### Otimizador Multivariado com Descida do Gradiente (C#)
* **Descrição:** Uma evolução da Regressão Linear, implementando **Descida do Gradiente Multivariado** e **Normalização de Features (Z-score)**.
* **Como Executar:** Siga os passos padrões.

</details>

---

## 📜 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE.md) para mais detalhes.