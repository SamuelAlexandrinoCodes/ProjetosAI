//1.  Coleta de Inteligencia
//Usamos Arrays para representar nossos vetores de dados
using System;
using System.Reflection.Metadata;

double[] horasEstudo_x = { 1, 2, 3, 4, 5 };
double[] notasObtidas_y = { 40, 50, 60, 70, 80 };

// 2. Parametros do Modelo e Hiperparametros
// Estes sao os valores que nosso modelo vai aprender
double m = 0.0; // Inclinacao Inicial
double b = 0.0; // Interceptacao Inicial
int epocas = 1000; // Quantas vezes vamos repetir o ciclo de treino
double taxaAprendizagem = 0.01; // Tamanho de cada atualizacao
int n = horasEstudo_x.Length; // Numero de pontos de dados

Console.WriteLine("--- INICIANDO TREINO DO MODELO DE REGRESSAO LINEAR COM DESCIDA DO GRADIENTE ---");
Console.WriteLine($"Parametros Iniciais: m = {m}, b = {b}");
Console.WriteLine($"Taxa de Aprendizagem: {taxaAprendizagem}, Epocas {epocas}");
Console.WriteLine("----------------------------------");

// 3. O loop de treinamento
for (int i = 0; i < epocas; i++)
{
    double gradiente_m_soma = 0;
    double gradiente_b_soma = 0;

    for (int j = 0; j < n; j++)
    {
        double x_atual = horasEstudo_x[j];
        double y_atual = notasObtidas_y[j];
        double previsao_y = m * x_atual + b;
        double erro = y_atual - previsao_y;

        gradiente_m_soma += x_atual * erro;
        gradiente_b_soma += erro;
    }
    double gradiente_m_final = (-2.0 / n) * gradiente_m_soma;
    double gradiente_b_final = (-2.0 / n) * gradiente_b_soma;
    m = m - taxaAprendizagem * gradiente_m_final;
    b = b - taxaAprendizagem * gradiente_b_final;

    if ((i+1) % 100 == 0)
    {
        double mse_total = 0;
        for (int j = 0; j < n; j++)
        {
            double previsao = m * horasEstudo_x[j] + b;
            mse_total += Math.Pow(notasObtidas_y[j] - previsao, 2);
        }
        Console.WriteLine($"Epoca {i + 1}: Erro(MSE) = {mse_total / n:F4}");
    }
}

// 4. Resultados Finais
Console.WriteLine("======================");
Console.WriteLine("--- TREINO CONCLUÍDO ---");
Console.WriteLine($"Parâmetros Finais Encontrados:");
Console.WriteLine($"Inclinação (m) = {m:F4}");
Console.WriteLine($"Intercepto (b) = {b:F4}");
Console.WriteLine($"\nEquação Final da Reta: y = {m:F2}x + {b:F2}");

// 5. Teste de fogo
double horasParaPrever = 6;
double notaPrevista = m * horasParaPrever + b;
Console.WriteLine($"\nPrevisão: Para {horasParaPrever} horas de estudo, a nota prevista é {notaPrevista:F2}");