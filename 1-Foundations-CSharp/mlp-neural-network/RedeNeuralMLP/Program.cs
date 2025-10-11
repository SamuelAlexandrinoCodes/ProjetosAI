// --- Missão 8.1: O Classificador de Risco de Diabetes (MLP) ---
// Objetivo: Construir uma Rede Neural Multi-Camada para um problema de diagnóstico médico.
// Esta é uma peça de portfólio que demonstra a capacidade de resolver um problema de classificação do mundo real com pré-processamento de dados.

using System;
using System.Collections.Generic;
using System.Linq;

// As classes Neuronio, Camada e RedeNeural são o nosso motor reutilizável e permanecem as mesmas.
public class Neuronio
{
    public double[] Pesos { get; set; }
    public double Bias { get; set; }
    public double EntradaBruta { get; private set; }
    public double Saida { get; private set; }
    public double Delta { get; set; }

    public Neuronio(int numeroDeEntradas)
    {
        var random = new Random(Guid.NewGuid().GetHashCode());
        Pesos = new double[numeroDeEntradas];
        for (int i = 0; i < numeroDeEntradas; i++)
        {
            Pesos[i] = random.NextDouble() * 2 - 1;
        }
        Bias = random.NextDouble() * 2 - 1;
    }

    private double FuncaoReLu(double z) => Math.Max(0, z);
    public double DerivadaReLu() => EntradaBruta > 0 ? 1 : 0;
    private double FuncaoSigmoide(double z) => 1.0 / (1.0 + Math.Exp(-z));
    public double DerivadaSigmoide() => Saida * (1 - Saida);

    public double Ativar(double[] entradas, bool usarReLu = true)
    {
        EntradaBruta = 0;
        for (int i = 0; i < Pesos.Length; i++)
        {
            EntradaBruta += Pesos[i] * entradas[i];
        }
        EntradaBruta += Bias;
        Saida = usarReLu ? FuncaoReLu(EntradaBruta) : FuncaoSigmoide(EntradaBruta);
        return Saida;
    }
}

public class Camada
{
    public List<Neuronio> Neuronios { get; set; }
    public Camada(int numeroDeNeuronios, int numeroDeEntradasPorNeuronio)
    {
        Neuronios = new List<Neuronio>();
        for (int i = 0; i < numeroDeNeuronios; i++)
        {
            Neuronios.Add(new Neuronio(numeroDeEntradasPorNeuronio));
        }
    }
}

public class RedeNeural
{
    public List<Camada> Camadas { get; set; }
    private double _taxaAprendizagem;

    public RedeNeural(double taxaAprendizagem = 0.1)
    {
        Camadas = new List<Camada>();
        _taxaAprendizagem = taxaAprendizagem;
    }

    public double[] Feedforward(double[] entradas)
    {
        var saidasCamadaAnterior = entradas;
        foreach (var camada in Camadas)
        {
            var saidasCamadaAtual = new List<double>();
            bool ehCamadaDeSaida = camada == Camadas.Last();
            foreach (var neuronio in camada.Neuronios)
            {
                saidasCamadaAtual.Add(neuronio.Ativar(saidasCamadaAnterior, !ehCamadaDeSaida));
            }
            saidasCamadaAnterior = saidasCamadaAtual.ToArray();
        }
        return saidasCamadaAnterior;
    }

    public void Backpropagate(double[] esperados)
    {
        for (int i = Camadas.Count - 1; i >= 0; i--)
        {
            var camada = Camadas[i];
            if (i == Camadas.Count - 1)
            {
                for (int j = 0; j < camada.Neuronios.Count; j++)
                {
                    var neuronio = camada.Neuronios[j];
                    double erro = neuronio.Saida - esperados[j];
                    neuronio.Delta = erro * neuronio.DerivadaSigmoide();
                }
            }
            else
            {
                for (int j = 0; j < camada.Neuronios.Count; j++)
                {
                    var neuronio = camada.Neuronios[j];
                    double erro = 0;
                    var camadaSeguinte = Camadas[i + 1];
                    foreach (var neuronioSeguinte in camadaSeguinte.Neuronios)
                    {
                        erro += neuronioSeguinte.Pesos[j] * neuronioSeguinte.Delta;
                    }
                    neuronio.Delta = erro * neuronio.DerivadaReLu();
                }
            }
        }
    }

    public void AtualizarPesos(double[] entradas)
    {
        for (int i = 0; i < Camadas.Count; i++)
        {
            var entradasParaCamada = (i == 0) ? entradas : Camadas[i - 1].Neuronios.Select(n => n.Saida).ToArray();
            foreach (var neuronio in Camadas[i].Neuronios)
            {
                for (int j = 0; j < entradasParaCamada.Length; j++)
                {
                    neuronio.Pesos[j] -= _taxaAprendizagem * neuronio.Delta * entradasParaCamada[j];
                }
                neuronio.Bias -= _taxaAprendizagem * neuronio.Delta;
            }
        }
    }
}

// O Quartel-General: agora orquestrando a missão de diagnóstico.
public class Programa
{
    public static void Main(string[] args)
    {
        Console.WriteLine("--- MISSÃO: CLASSIFICADOR DE RISCO DE DIABETES (MLP) ---");

        // --- FASE 1: PREPARAÇÃO DO TERRENO ---
        // Inteligência de Campo: 3 características para prever 2 classes (risco ou não)
        // Entradas: {Glicose, IMC, Idade}
        double[][] entradas_x_bruto =
        {
            // Sem Risco (Classe 0)
            new double[] {90, 22, 25}, new double[] {105, 24, 30},
            new double[] {85, 21, 22}, new double[] {110, 26, 35},
            // Com Risco (Classe 1)
            new double[] {145, 31, 45}, new double[] {160, 35, 55},
            new double[] {180, 38, 60}, new double[] {150, 33, 50}
        };
        double[][] resultados_y =
        {
            new double[] {0}, new double[] {0}, new double[] {0}, new double[] {0},
            new double[] {1}, new double[] {1}, new double[] {1}, new double[] {1}
        };

        // Normalização dos Dados (Z-score)
        var medias = new double[3];
        var desviosPadrao = new double[3];
        var entradas_x_normalizado = new double[entradas_x_bruto.Length][];

        for (int i = 0; i < 3; i++)
        {
            medias[i] = entradas_x_bruto.Average(row => row[i]);
            double somaDosQuadrados = entradas_x_bruto.Sum(row => Math.Pow(row[i] - medias[i], 2));
            desviosPadrao[i] = Math.Sqrt(somaDosQuadrados / entradas_x_bruto.Length);
        }

        for (int i = 0; i < entradas_x_bruto.Length; i++)
        {
            entradas_x_normalizado[i] = new double[3];
            for (int j = 0; j < 3; j++)
            {
                entradas_x_normalizado[i][j] = (entradas_x_bruto[i][j] - medias[j]) / desviosPadrao[j];
            }
        }
        Console.WriteLine("--- Dados normalizados com sucesso. ----");

        // --- FASE 2: TREINAMENTO ---
        var rede = new RedeNeural(0.1);
        rede.Camadas.Add(new Camada(5, 3)); // Camada oculta: 5 neurónios, 3 entradas
        rede.Camadas.Add(new Camada(1, 5)); // Camada de saída: 1 neurónio, 5 entradas

        int epocas = 5000;
        for (int i = 0; i < epocas; i++)
        {
            for (int j = 0; j < entradas_x_normalizado.Length; j++)
            {
                rede.Feedforward(entradas_x_normalizado[j]);
                rede.Backpropagate(resultados_y[j]);
                rede.AtualizarPesos(entradas_x_normalizado[j]);
            }
        }
        Console.WriteLine("\n--- TREINO CONCLUÍDO. ---");

        // --- FASE 3: TESTE DE FOGO ---
        Console.WriteLine("\n--- TESTANDO O ESPECIALISTA EM DIAGNÓSTICO... ---");

        // Teste com um paciente de baixo risco
        var pacienteBaixoRisco = new double[] { 100, 25, 32 };
        var pacienteBaixoRisco_norm = new double[3];
        for (int i = 0; i < 3; i++) { pacienteBaixoRisco_norm[i] = (pacienteBaixoRisco[i] - medias[i]) / desviosPadrao[i]; }
        var previsaoBaixo = rede.Feedforward(pacienteBaixoRisco_norm);
        Console.WriteLine($"Paciente Baixo Risco: [Glic:{pacienteBaixoRisco[0]}, IMC:{pacienteBaixoRisco[1]}, Idade:{pacienteBaixoRisco[2]}], Previsão: {previsaoBaixo[0]:P2} de Risco -> Veredicto: {(previsaoBaixo[0] < 0.5 ? "Sem Risco" : "Com Risco")}");

        // Teste com um paciente de alto risco
        var pacienteAltoRisco = new double[] { 165, 36, 58 };
        var pacienteAltoRisco_norm = new double[3];
        for (int i = 0; i < 3; i++) { pacienteAltoRisco_norm[i] = (pacienteAltoRisco[i] - medias[i]) / desviosPadrao[i]; }
        var previsaoAlto = rede.Feedforward(pacienteAltoRisco_norm);
        Console.WriteLine($"Paciente Alto Risco: [Glic:{pacienteAltoRisco[0]}, IMC:{pacienteAltoRisco[1]}, Idade:{pacienteAltoRisco[2]}], Previsão: {previsaoAlto[0]:P2} de Risco -> Veredicto: {(previsaoAlto[0] < 0.5 ? "Sem Risco" : "Com Risco")}");

        // Teste com um paciente de médio risco (Sua excelente adição, agora corrigida)
        var pacienteMedioRisco = new double[] { 125, 32, 35 };
        var pacienteMedioRisco_norm = new double[3];
        for (int i = 0; i < 3; i++) { pacienteMedioRisco_norm[i] = (pacienteMedioRisco[i] - medias[i]) / desviosPadrao[i]; }
        var previsaoMedio = rede.Feedforward(pacienteMedioRisco_norm);
        Console.WriteLine($"Paciente Médio Risco: [Glic:{pacienteMedioRisco[0]}, IMC:{pacienteMedioRisco[1]}, Idade:{pacienteMedioRisco[2]}], Previsão: {previsaoMedio[0]:P2} de Risco -> Veredicto: {(previsaoMedio[0] < 0.5 ? "Sem Risco" : "Com Risco")}");
    }
}

