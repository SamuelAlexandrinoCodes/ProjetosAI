// --- Missão 9.4: A CNN de Elite (Múltiplos Filtros) ---
// Objetivo: Aumentar o poder da CNN com múltiplos filtros para resolver problemas não-lineares.

using System;
using System.Collections.Generic;
using System.Linq;

// --- FUNÇÕES AUXILIARES E DE ATIVAÇÃO ---
public static class Utils
{
    public static double ReLU(double x) => Math.Max(0, x);
    public static double DerivadaReLU(double x) => x > 0 ? 1 : 0;
    public static double FuncaoSigmoide(double z) => 1.0 / (1.0 + Math.Exp(-z));
    public static double DerivadaSigmoide(double sig) => sig * (1 - sig);
}

// --- AS UNIDADES DE COMBATE (AS CAMADAS) ---
public interface ICamada
{
    object Feedforward(object entrada);
    object Backpropagate(object gradienteSaida, double taxaAprendizagem);
}

public class CamadaConvolucional : ICamada
{
    public List<double[,]> Filtros { get; set; }
    public int NumeroDeFiltros { get; }
    private double[,] _ultimaEntrada = new double[0,0];
    private List<double[,]> _ultimasEntradasBrutas = new List<double[,]>();

    public CamadaConvolucional(int numeroDeFiltros, int tamanhoFiltro)
    {
        NumeroDeFiltros = numeroDeFiltros;
        Filtros = new List<double[,]>();
        // Usar um único Random para evitar o problema da semente de tempo
        var random = new Random(); 
        for (int i = 0; i < numeroDeFiltros; i++)
        {
            var filtro = new double[tamanhoFiltro, tamanhoFiltro];
            for (int y = 0; y < tamanhoFiltro; y++)
            for (int x = 0; x < tamanhoFiltro; x++)
                // A inicialização aleatória quebra a simetria entre os filtros
                filtro[y, x] = random.NextDouble() * 2 - 1;
            Filtros.Add(filtro);
        }
    }

    public object Feedforward(object entradaObj)
    {
        _ultimaEntrada = (double[,])entradaObj;
        _ultimasEntradasBrutas.Clear();
        var saidas = new List<double[,]>();

        foreach (var filtro in Filtros)
        {
            int alturaSaida = _ultimaEntrada.GetLength(0) - filtro.GetLength(0) + 1;
            int larguraSaida = _ultimaEntrada.GetLength(1) - filtro.GetLength(1) + 1;
            var saida = new double[alturaSaida, larguraSaida];
            var entradaBruta = new double[alturaSaida, larguraSaida];

            for (int y = 0; y < alturaSaida; y++)
            for (int x = 0; x < larguraSaida; x++)
            {
                double soma = 0;
                for (int fy = 0; fy < filtro.GetLength(0); fy++)
                for (int fx = 0; fx < filtro.GetLength(1); fx++)
                {
                    soma += _ultimaEntrada[y + fy, x + fx] * filtro[fy, fx];
                }
                entradaBruta[y, x] = soma;
                saida[y, x] = Utils.ReLU(soma);
            }
            saidas.Add(saida);
            _ultimasEntradasBrutas.Add(entradaBruta);
        }
        return saidas;
    }

    public object Backpropagate(object gradienteSaidaObj, double taxaAprendizagem)
    {
        var gradientesSaida = (List<double[,]>)gradienteSaidaObj;
        
        for(int f = 0; f < NumeroDeFiltros; f++)
        {
            var filtro = Filtros[f];
            var gradienteSaidaFiltro = gradientesSaida[f];
            var ultimaEntradaBrutaFiltro = _ultimasEntradasBrutas[f];
            var gradienteFiltro = new double[filtro.GetLength(0), filtro.GetLength(1)];
            var delta = new double[gradienteSaidaFiltro.GetLength(0), gradienteSaidaFiltro.GetLength(1)];

            for(int y = 0; y < delta.GetLength(0); y++)
            for(int x = 0; x < delta.GetLength(1); x++)
            {
                delta[y, x] = gradienteSaidaFiltro[y, x] * Utils.DerivadaReLU(ultimaEntradaBrutaFiltro[y, x]);
            }

            for (int y = 0; y < delta.GetLength(0); y++)
            for (int x = 0; x < delta.GetLength(1); x++)
            {
                for (int fy = 0; fy < filtro.GetLength(0); fy++)
                for (int fx = 0; fx < filtro.GetLength(1); fx++)
                {
                    gradienteFiltro[fy, fx] += _ultimaEntrada[y + fy, x + fx] * delta[y, x];
                }
            }
        
            for (int i = 0; i < filtro.GetLength(0); i++)
            for (int j = 0; j < filtro.GetLength(1); j++)
                filtro[i, j] -= taxaAprendizagem * gradienteFiltro[i, j];
        }
        return null;
    }
}

public class CamadaDeAchatamento : ICamada
{
    private List<(int depth, int height, int width)> _dimensoesEntrada = new List<(int,int,int)>();

    public object Feedforward(object entradaObj)
    {
        var entradas = (List<double[,]>)entradaObj;
        _dimensoesEntrada.Clear();
        var listaAchatada = new List<double>();
        foreach(var matriz in entradas)
        {
            _dimensoesEntrada.Add((0, matriz.GetLength(0), matriz.GetLength(1)));
            listaAchatada.AddRange(matriz.Cast<double>());
        }
        return listaAchatada.ToArray();
    }

    public object Backpropagate(object gradienteSaidaObj, double taxaAprendizagem)
    {
        var gradienteSaida = (double[])gradienteSaidaObj;
        var gradientesEntrada = new List<double[,]>();
        int indiceAtual = 0;
        foreach(var dim in _dimensoesEntrada)
        {
            var gradienteMatriz = new double[dim.height, dim.width];
            for(int i = 0; i < dim.height; i++)
            for(int j = 0; j < dim.width; j++)
            {
                gradienteMatriz[i, j] = gradienteSaida[indiceAtual++];
            }
            gradientesEntrada.Add(gradienteMatriz);
        }
        return gradientesEntrada;
    }
}

public class CamadaTotalmenteConectada : ICamada
{
    public double[] Pesos { get; set; }
    public double Bias { get; set; }
    private double[] _ultimaEntrada = new double[0];
    private double _ultimaSaida;

    public CamadaTotalmenteConectada(int numeroDeEntradas, int numeroDeSaidas = 1)
    {
        Pesos = new double[numeroDeEntradas];
        var random = new Random(Guid.NewGuid().GetHashCode());
        for (int i = 0; i < numeroDeEntradas; i++)
            Pesos[i] = random.NextDouble() * 2 - 1;
        Bias = random.NextDouble() * 2 - 1;
    }

    public object Feedforward(object entradaObj)
    {
        _ultimaEntrada = (double[])entradaObj;
        double somaPonderada = _ultimaEntrada.Zip(Pesos, (a, b) => a * b).Sum() + Bias;
        _ultimaSaida = Utils.FuncaoSigmoide(somaPonderada);
        return _ultimaSaida;
    }

    public object Backpropagate(object gradienteSaidaObj, double taxaAprendizagem)
    {
        double erro = (double)gradienteSaidaObj;
        double delta = erro * Utils.DerivadaSigmoide(_ultimaSaida);
        var gradienteEntrada = new double[Pesos.Length];

        for (int i = 0; i < Pesos.Length; i++)
        {
            gradienteEntrada[i] = Pesos[i] * delta;
            Pesos[i] -= taxaAprendizagem * delta * _ultimaEntrada[i];
        }
        Bias -= taxaAprendizagem * delta;

        return gradienteEntrada;
    }
}

// --- O QUARTEL-GENERAL (A ORQUESTRAÇÃO) ---
public class Programa
{
    public static void Main(string[] args)
    {
        Console.WriteLine("--- FORJANDO E TREINANDO UMA CNN DE ELITE PARA O PROBLEMA XOR ---");

        var entradas_x = new double[][,]
        {
            new double[,] {{0, 0}, {0, 0}},
            new double[,] {{0, 1}, {0, 0}},
            new double[,] {{1, 0}, {0, 0}},
            new double[,] {{1, 1}, {0, 0}}
        };
        var resultados_y = new double[] { 0, 1, 1, 0 };

        int numeroDeFiltros = 8; // Aumentamos o esquadrão
        int tamanhoFiltro = 2;
        var camadas = new ICamada[]
        {
            new CamadaConvolucional(numeroDeFiltros, tamanhoFiltro),
            new CamadaDeAchatamento(),
            new CamadaTotalmenteConectada(numeroDeFiltros, 1)
        };
        
        int epocas = 5000; // Aumentamos o tempo de treino
        double taxaAprendizagem = 0.1;

        for (int i = 0; i < epocas; i++)
        {
            double erroTotalEpoca = 0;
            for (int j = 0; j < entradas_x.Length; j++)
            {
                object saidaAtual = entradas_x[j];
                foreach (var camada in camadas)
                {
                    saidaAtual = camada.Feedforward(saidaAtual);
                }
                double previsao = (double)saidaAtual;
                
                erroTotalEpoca += Math.Pow(resultados_y[j] - previsao, 2);

                object gradiente = previsao - resultados_y[j];
                for (int k = camadas.Length - 1; k >= 0; k--)
                {
                    gradiente = camadas[k].Backpropagate(gradiente, taxaAprendizagem);
                }
            }
            if ((i + 1) % 500 == 0)
            {
                Console.WriteLine($"Época {i + 1}, Erro Total: {erroTotalEpoca / entradas_x.Length:F4}");
            }
        }

        Console.WriteLine("\n--- TREINO CONCLUÍDO. TESTANDO A CNN DE ELITE... ---");
        for (int i = 0; i < entradas_x.Length; i++)
        {
            object saidaAtual = entradas_x[i];
            foreach (var camada in camadas)
            {
                saidaAtual = camada.Feedforward(saidaAtual);
            }
            double previsao = (double)saidaAtual;

            string entradaStr = $"[{entradas_x[i][0,0]},{entradas_x[i][0,1]}]";
            Console.WriteLine($"Entrada: {entradaStr}, Previsão: {previsao:F4} (Esperado: {resultados_y[i]})");
        }
    }
}

