// --- Missão 6.1 (Recontextualizada): O Classificador de Diagnóstico Médico ---
// Objetivo: Usar nossa tática de treino em lote para prever a malignidade de um tumor.

using System;

// A classe Neuronio permanece a mesma. É o nosso bloco de construção universal.
public class Neuronio
{
    public double[] Pesos { get; set; }
    public double Bias { get; set; }

    public Neuronio(int numeroDeEntradas)
    {
        var random = new Random();
        Pesos = new double[numeroDeEntradas];
        for (int i = 0; i < numeroDeEntradas; i++)
        {
            Pesos[i] = random.NextDouble() - 0.5;
        }
        Bias = random.NextDouble() - 0.5;
    }

    private double FuncaoSigmoide(double z) => 1.0 / (1.0 + Math.Exp(-z));

    public double Ativar(double[] entradas)
    {
        double poderBruto = 0;
        for (int i = 0; i < Pesos.Length; i++)
        {
            poderBruto += Pesos[i] * entradas[i];
        }
        poderBruto += Bias;

        return FuncaoSigmoide(poderBruto);
    }
}

// Classe principal que agora implementa a missão de diagnóstico.
public class Programa
{
    public static void Main(string[] args)
    {
        Console.WriteLine("--- FORJANDO UM NEURÓNIO PARA MISSÃO DE DIAGNÓSTICO MÉDICO (TÁTICA EM LOTE) ---");

        // 1. Inteligência de Campo (Novos Dados de Tumores)
        double[] tamanhoTumor_x = { 1.5, 2.0, 3.2, 5.0, 6.1, 8.5 }; // Tamanho em cm
        double[] resultado_y = { 0, 0, 0, 1, 1, 1 };   // 0 = Benigno, 1 = Maligno
        int n_amostras = resultado_y.Length;

        var neuronioClassificador = new Neuronio(1);

        // 2. Hiperparâmetros do Treino
        int epocas = 10000;
        double taxaAprendizagem = 0.1;

        Console.WriteLine("--- INICIANDO TREINO DO NEURÓNIO (BATCH GRADIENT DESCENT) ---");

        // 3. O Loop de Treinamento (A LÓGICA É IDÊNTICA)
        for (int i = 0; i < epocas; i++)
        {
            double gradiente_peso_soma = 0;
            double gradiente_bias_soma = 0;

            for (int j = 0; j < n_amostras; j++)
            {
                double[] entrada = { tamanhoTumor_x[j] };
                double real = resultado_y[j];

                double previsao = neuronioClassificador.Ativar(entrada);
                double erro = previsao - real;

                gradiente_peso_soma += erro * entrada[0];
                gradiente_bias_soma += erro;
            }

            double gradiente_peso_medio = gradiente_peso_soma / n_amostras;
            double gradiente_bias_medio = gradiente_bias_soma / n_amostras;

            neuronioClassificador.Pesos[0] -= taxaAprendizagem * gradiente_peso_medio;
            neuronioClassificador.Bias -= taxaAprendizagem * gradiente_bias_medio;
        }

        Console.WriteLine("--- TREINO CONCLUÍDO ---");
        Console.WriteLine("\nParâmetros Finais do Neurónio Aprendidos:");
        Console.WriteLine($"Peso (w1 / Influência do Tamanho) = {neuronioClassificador.Pesos[0]:F4}");
        Console.WriteLine($"Bias (b / Limiar de Ativação) = {neuronioClassificador.Bias:F4}");

        // 4. Teste de Fogo (Novos Casos de Diagnóstico)
        Console.WriteLine("\n--- TESTE DE DIAGNÓSTICO ---");

        double[] teste1_entrada = { 2.5 }; // Um tumor pequeno
        double previsao1 = neuronioClassificador.Ativar(teste1_entrada);
        Console.WriteLine($"Diagnóstico para tumor de {teste1_entrada[0]} cm: {previsao1 * 100:F2}% de chance de ser maligno. Veredicto: {(previsao1 > 0.5 ? "Maligno" : "Benigno")}");

        double[] teste2_entrada = { 5.5 }; // Um tumor grande
        double previsao2 = neuronioClassificador.Ativar(teste2_entrada);
        Console.WriteLine($"Diagnóstico para tumor de {teste2_entrada[0]} cm: {previsao2 * 100:F2}% de chance de ser maligno. Veredicto: {(previsao2 > 0.5 ? "Maligno" : "Benigno")}");

        //Usar uma instância de Random para gerar o valor aleatório
        var random = new Random();
        double[] teste3_entrada = { random.NextDouble() * (9.0 - 0.5) + 0.5 }; // Um tumor aleatório entre 0.5 e 9.0 cm
        double previsao3 = neuronioClassificador.Ativar(teste3_entrada);
        Console.WriteLine($"Diagnóstico para tumor aleatório de {teste3_entrada[0]:F2} cm: {previsao3 * 100:F2}% de chance de ser maligno. Veredicto: {(previsao3 > 0.5 ? "Maligno" : "Benigno")}");
    }
}

