// --- Missão 7.0: A Fábrica de Mini-Lotes ---
// Objetivo: Implementar um classificador usando a tática de treino Mini-Batch Gradient Descent.

using System;
using System.Collections.Generic;
using System.Linq;

// A classe Neuronio permanece a mesma. É a nossa unidade de combate padrão.
public class Neuronio
{
    public double[] Pesos { get; set; }
    public double Bias { get; set; }
    private Random _random = new Random();

    public Neuronio(int numeroDeEntradas)
    {
        Pesos = new double[numeroDeEntradas];
        for (int i = 0; i < numeroDeEntradas; i++)
        {
            Pesos[i] = _random.NextDouble() - 0.5;
        }
        Bias = _random.NextDouble() - 0.5;
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

// Classe principal que irá orquestrar a nossa nova linha de produção.
public class Programa
{
    public static void Main(string[] args)
    {
        Console.WriteLine("--- INICIANDO A CONSTRUÇÃO DA FÁBRICA DE MINI-LOTES ---");

        // 1. Inteligência de Campo (Dataset Iris Simplificado)
        // Cada linha é uma flor: {Comprimento Sepala, Largura Sepala}
        double[][] entradas_x =
        {
            new double[] {5.1, 3.5}, new double[] {4.9, 3.0}, new double[] {4.7, 3.2}, // Iris Setosa (0)
            new double[] {5.0, 3.6}, new double[] {5.4, 3.9},
            new double[] {7.0, 3.2}, new double[] {6.4, 3.2}, new double[] {6.9, 3.1}, // Iris Versicolor (1)
            new double[] {6.5, 2.8}, new double[] {5.7, 2.8}
        };

        // As classificações reais correspondentes
        double[] resultados_y = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };

        Console.WriteLine($"Total de amostras de treino: {entradas_x.Length}");

        // 2. Hiperparâmetros e Forja do Neurónio
        int tamanhoLote = 2; // Tamanho do nosso esquadrão
        int epocas = 10000;
        double taxaAprendizagem = 0.1;

        // O nosso neurónio agora tem 2 entradas (Comprimento e Largura da Sépala)
        var neuronioClassificador = new Neuronio(2);
        var random = new Random();

        Console.WriteLine($"\n--- INICIANDO TREINO (MINI-BATCH GD / TAMANHO DO LOTE: {tamanhoLote}) ---");

        // 3. O Loop de Treino Principal (Épocas)
        for (int i = 0; i < epocas; i++)
        {
            // --- Fase de Embaralhamento ---
            // Criamos uma lista de índices (0, 1, 2, ... 9) e a embaralhamos.
            var indices = Enumerable.Range(0, entradas_x.Length).OrderBy(c => random.Next()).ToList();

            // --- Loop sobre os Mini-Lotes ---
            for (int j = 0; j < entradas_x.Length; j += tamanhoLote)
            {
                // Pegamos um esquadrão (mini-lote) de índices
                var indicesDoLote = indices.Skip(j).Take(tamanhoLote).ToList();

                // Se o último lote for menor que o tamanho do lote, não faz mal
                if (!indicesDoLote.Any()) continue;

                // Acumuladores de gradiente PARA ESTE LOTE
                double gradiente_peso_soma_0 = 0;
                double gradiente_peso_soma_1 = 0;
                double gradiente_bias_soma = 0;

                // --- Batalha do Mini-Lote ---
                // Itera sobre cada soldado no esquadrão
                foreach (var indice in indicesDoLote)
                {
                    var entrada = entradas_x[indice];
                    var real = resultados_y[indice];

                    var previsao = neuronioClassificador.Ativar(entrada);
                    var erro = previsao - real;

                    // Acumula os gradientes
                    gradiente_peso_soma_0 += erro * entrada[0];
                    gradiente_peso_soma_1 += erro * entrada[1];
                    gradiente_bias_soma += erro;
                }

                // --- Atualização Pós-Batalha ---
                // Calcula o gradiente médio para o lote e atualiza os pesos
                double gradiente_peso_medio_0 = gradiente_peso_soma_0 / indicesDoLote.Count;
                double gradiente_peso_medio_1 = gradiente_peso_soma_1 / indicesDoLote.Count;
                double gradiente_bias_medio = gradiente_bias_soma / indicesDoLote.Count;

                neuronioClassificador.Pesos[0] -= taxaAprendizagem * gradiente_peso_medio_0;
                neuronioClassificador.Pesos[1] -= taxaAprendizagem * gradiente_peso_medio_1;
                neuronioClassificador.Bias -= taxaAprendizagem * gradiente_bias_medio;
            }
        }

        Console.WriteLine("--- TREINO CONCLUÍDO ---");
        Console.WriteLine("\nParâmetros Finais do Neurónio Aprendidos:");
        Console.WriteLine($"Peso w1 (Comprimento Sépala) = {neuronioClassificador.Pesos[0]:F4}");
        Console.WriteLine($"Peso w2 (Largura Sépala) = {neuronioClassificador.Pesos[1]:F4}");
        Console.WriteLine($"Bias (b) = {neuronioClassificador.Bias:F4}");

        // 4. Teste de Fogo
        Console.WriteLine("\n--- TESTE DE CLASSIFICAÇÃO ---");

        // Uma flor com características de Setosa (pequena e larga)
        double[] teste_setosa = { 5.0, 3.5 };
        double previsao_setosa = neuronioClassificador.Ativar(teste_setosa);
        Console.WriteLine($"Diagnóstico para flor [{teste_setosa[0]}, {teste_setosa[1]}]: {previsao_setosa * 100:F2}% de chance de ser Versicolor. Veredicto: {(previsao_setosa > 0.5 ? "Versicolor" : "Setosa")}");

        // Uma flor com características de Versicolor (grande e estreita)
        double[] teste_versicolor = { 6.5, 3.0 };
        double previsao_versicolor = neuronioClassificador.Ativar(teste_versicolor);
        Console.WriteLine($"Diagnóstico para flor [{teste_versicolor[0]}, {teste_versicolor[1]}]: {previsao_versicolor * 100:F2}% de chance de ser Versicolor. Veredicto: {(previsao_versicolor > 0.5 ? "Versicolor" : "Setosa")}");
    }
}

