// --- MISSAO 4.3: O Otimizador Multivariado ---
// Autor: Samuel Alexandrino de Oliveira
// Data: 2024-06-20
// Descrição: Este programa implementa um otimizador multivariado para ajustar os parâmetros de um modelo preditivo, minimizando o erro entre as previsões do modelo e os dados reais.

// FASE 1: Preparacao do Ambiente

//1.1 Coleta de Dados
double[] horasEstudo_x1 = { 5, 8, 2, 10, 6 };
double[] faltas_x2 = { 4, 1, 5, 0, 2 };
double[] notaAnterior_x3 = { 70, 85, 60, 90, 75 };
double[] notaFinal_y = { 65, 90, 50, 98, 75 }; //alvo

int n_amostras = notaFinal_y.Length;
int n_features = 3;

var x_normalizado = new double[n_amostras, n_features];

var medias = new double[n_features];
var desviosPadrao = new double[n_features];

//1.2 Normalizacao dos Dados
Console.WriteLine("--- FASE 1: Normalizando os dados... ---");

var dadosBrutos = new List <double[]> { horasEstudo_x1, faltas_x2, notaAnterior_x3 };

for (int i = 0; i < n_features; i++)
{
    medias[i] = dadosBrutos[i].Average();
    double somaDosQuadrados = dadosBrutos[i].Select(val => (val - medias[i]) * (val - medias[i])).Sum();
    desviosPadrao[i] = Math.Sqrt(somaDosQuadrados / n_amostras);

    Console.WriteLine($"Feature {i + 1}: Média = {medias[i]:F2}, Desvio Padrão = {desviosPadrao[i]:F2}");
    
    // Normalização Z-score
    for (int j = 0; j < n_amostras; j++)
    {
        x_normalizado[j, i] = (dadosBrutos[i][j] - medias[i]) / desviosPadrao[i];
    }
}
Console.WriteLine("--- Dados normalizados com sucesso. ----");

// FASE 2: TREINAMENTO DO MODELO

Console.WriteLine("--- FASE 2: Treinando o modelo... ---");

//2.1 Parametros do Modelo

var pesos_w = new double[n_features];
double b = 0.0;

int epocas = 100000;
double taxaAprendizagem = 0.1;

Console.WriteLine("\n --- FASE 2: Iniciando o treinamento do Modelo Multivariado --- \n");

for (int i = 0; i < epocas; i++)
{
    var gradientes_w = new double[n_features];
    double gradiente_b = 0.0;

    for (int j = 0; j < n_amostras; j++)
    {
        double x1 = x_normalizado[j, 0];
        double x2 = x_normalizado[j, 1];
        double x3 = x_normalizado[j, 2];

        double previsao = (pesos_w[0] * x1) + (pesos_w[1] * x2) + (pesos_w[2] * x3) + b;

        double erro = notaFinal_y[j] - previsao;

        gradientes_w[0] += x1 * erro;
        gradientes_w[1] += x2 * erro;
        gradientes_w[2] += x3 * erro;
        gradiente_b += erro;
    }

    for (int k = 0; k < n_features; k++)
    {
        pesos_w[k] = pesos_w[k] - taxaAprendizagem * ((-2.0 / n_amostras) * gradientes_w[k]);

    }
    b = b - taxaAprendizagem * ((-2.0 / n_amostras) * gradiente_b);
}

Console.WriteLine("--- TREINO CONCLUÍDO ---");
Console.WriteLine("\nParâmetros Finais Encontrados:");
Console.WriteLine($"Peso w1 (Horas Estudo) = {pesos_w[0]:F4}");
Console.WriteLine($"Peso w2 (Faltas) = {pesos_w[1]:F4}");
Console.WriteLine($"Peso w3 (Nota Anterior) = {pesos_w[2]:F4}");
Console.WriteLine($"Intercepto (b) = {b:F4}");

// FASE 3: PREVISAO
Console.WriteLine("\n--- FASE 3: Teste de Fogo com Novo Aluno ---");

// 3.1 Novo Aluno
double[] novoAluno_dadosBrutos = { 12, 0, 95 }; // 7 horas de estudo, 3 faltas, nota anterior 80
var novoAluno_normalizado = new double[n_features];

Console.WriteLine($"Dados do novo aluno: Horas={novoAluno_dadosBrutos[0]}, Faltas={novoAluno_dadosBrutos[1]}, Nota Ant.={novoAluno_dadosBrutos[2]}");

for (int i = 0; i < n_features; i++)
{
    novoAluno_normalizado[i] = (novoAluno_dadosBrutos[i] - medias[i]) / desviosPadrao[i];
}

double notaPrevista = (pesos_w[0] * novoAluno_normalizado[0]) +
                      (pesos_w[1] * novoAluno_normalizado[1]) +
                      (pesos_w[2] * novoAluno_normalizado[2]) + 
                      b;

Console.WriteLine($"Nota Final Prevista para o Novo Aluno: {notaPrevista:F2}");