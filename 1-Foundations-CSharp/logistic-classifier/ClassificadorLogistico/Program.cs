// --- Missão 5.1: O Classificador Logístico em Código ---
// Objetivo: Forjar um modelo que aprende a fronteira de decisão para um problema de classificação binária.

// --- FASE 1: PREPARAÇÃO DO CAMPO DE BATALHA ---

// 1. Inteligência de Campo (Nossos Dados de Treino)
double[] horasEstudo_x = { 1, 2, 4, 5, 6, 7, 8 };
double[] resultado_y = { 0, 0, 0, 1, 1, 1, 1 }; // 0 = Reprovado, 1 = Aprovado

int n_amostras = horasEstudo_x.Length;

// 2. Parâmetros do Modelo e Hiperparâmetros
// Estes são os parâmetros da nossa fronteira de decisão (z = mx + b)
double m = 0.0;
double b = 0.0;

// Hiperparâmetros que controlam o nosso treino
int epocas = 10000;
double taxaAprendizagem = 0.1; // Uma taxa um pouco mais agressiva para este problema

// 3. A Nossa Arma Principal: A Prensa de Probabilidades
static double FuncaoSigmoide(double z)
{
    return 1.0 / (1.0 + Math.Exp(-z));
}

Console.WriteLine("--- INICIANDO TREINO DO CLASSIFICADOR LOGÍSTICO ---");
Console.WriteLine($"Parâmetros Iniciais: m = {m:F4}, b = {b:F4}");
Console.WriteLine("-----------------------------------------------------");

// --- FASE 2: O PROCESSO DE TREINAMENTO (A FORJA) ---
for (int i = 0; i < epocas; i++)
{
    // A cada época, reiniciamos a nossa "bússola"
    double gradiente_m_soma = 0;
    double gradiente_b_soma = 0;

    // A nossa linha de montagem processa cada aluno
    for (int j = 0; j < n_amostras; j++)
    {
        double x = horasEstudo_x[j];
        double y = resultado_y[j];

        // Estágio 1: Calcular o "Poder Bruto"
        double z = m * x + b;

        // Estágio 2: Passar pela "Prensa de Probabilidades"
        double p = FuncaoSigmoide(z);

        // Estágio 3: Calcular o Erro (a munição para o nosso gradiente)
        double erro = p - y;

        // Acumular os gradientes para esta época
        gradiente_m_soma += erro * x;
        gradiente_b_soma += erro;
    }

    // O Passo de Aprendizagem: Atualizar os parâmetros na direção oposta ao gradiente médio
    m = m - taxaAprendizagem * (gradiente_m_soma / n_amostras);
    b = b - taxaAprendizagem * (gradiente_b_soma / n_amostras);
}

Console.WriteLine("--- TREINO CONCLUÍDO ---");
Console.WriteLine("\nParâmetros Finais da Fronteira de Decisão Encontrados:");
Console.WriteLine($"Inclinação (m) = {m:F4}");
Console.WriteLine($"Intercepto (b) = {b:F4}");
Console.WriteLine($"\nEquação da Fronteira: {m:F2}x + {b:F2} = 0");

// --- FASE 3: TESTE DE FOGO ---
Console.WriteLine("\n--- TESTANDO O MODELO TREINADO ---");

// Cenário 1: Um aluno que estudou pouco (3 horas)
double horas_teste_1 = 3;
double z1 = m * horas_teste_1 + b;
double prob1 = FuncaoSigmoide(z1);
Console.WriteLine($"Previsão para {horas_teste_1} horas de estudo:");
Console.WriteLine($"  -> Probabilidade de Aprovação: {prob1:P2}"); // P2 formata como percentagem
Console.WriteLine($"  -> Veredito: {(prob1 > 0.5 ? "PROVAVELMENTE APROVADO" : "PROVAVELMENTE REPROVADO")}");

// Cenário 2: Um aluno que estudou muito (6 horas)
double horas_teste_2 = 6;
double z2 = m * horas_teste_2 + b;
double prob2 = FuncaoSigmoide(z2);
Console.WriteLine($"\nPrevisão para {horas_teste_2} horas de estudo:");
Console.WriteLine($"  -> Probabilidade de Aprovação: {prob2:P2}");
Console.WriteLine($"  -> Veredito: {(prob2 > 0.5 ? "PROVAVELMENTE APROVADO" : "PROVAVELMENTE REPROVADO")}");
