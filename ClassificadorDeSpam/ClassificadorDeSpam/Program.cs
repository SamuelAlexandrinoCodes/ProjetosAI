string[] spamFrases = {"clique aqui oferta imperdivel", "ganhe dinheiro rápido", "trabalho em casa", "promoção exclusiva", "compre agora e economize"};
string[] hamFrases = {"reunião agendada para amanhã", "relatório de vendas do mês", "atualização do projeto", "convite para o evento", "notícias da empresa"};

var spamCounts = new Dictionary<string, int>();
var hamCounts = new Dictionary<string, int>();

Console.WriteLine("Treinando o classificador de spam...");
foreach (var frase in spamFrases)
{
    var palavras = frase.Split(' ');
    foreach (var palavra in palavras)
    {
        if (spamCounts.ContainsKey(palavra))
            spamCounts[palavra]++;
        else
            spamCounts[palavra] = 1;
    }
}
Console.WriteLine("Treinando dados com Ham...");
foreach (var frase in hamFrases)
{
    var palavras = frase.Split(' ');
    foreach (var palavra in palavras)
    {
        if (hamCounts.ContainsKey(palavra))
            hamCounts[palavra]++;
        else
            hamCounts[palavra] = 1;
    }
}
Console.WriteLine("\n--- Frequencia de Palavras em Spam ---");
foreach (var par in spamCounts)
{
    Console.WriteLine($"Palavra: `{par.Key}`, Contagem: {par.Value}`");
}
Console.WriteLine("\n--- Frequencia de Palavras em Ham ---");
foreach (var par in hamCounts)
{
    Console.WriteLine($"Palavra: `{par.Key}`, Contagem: {par.Value}`");
}
// --   Fim do Treinamento  -- 
// --- Fase 2: Classificação ---
Console.WriteLine("\n--- INICIANDO TESTE DE CLASSIFICAÇÃO ---");

string[] frasesTeste = {
    "clique aqui para ganhar dinheiro",
    "relatório de vendas atualizado",
    "promoção exclusiva para você",
    "reunião sobre o projeto amanhã"
};

Console.WriteLine($"Analisando as frases: '{frasesTeste}");
// Loop externo: analisa uma frase de teste por vez
foreach (var fraseAtual in frasesTeste)
{
    Console.WriteLine($"\n---- ANALISANDO A FRASE: '{fraseAtual}'-----");
    int spamScore = 0;
    int hamScore = 0;

    var palavrasTeste = fraseAtual.Split(' ');


    foreach (var palavra in palavrasTeste)
    {
        if (spamCounts.ContainsKey(palavra))
            spamScore += spamCounts[palavra];
        if (hamCounts.ContainsKey(palavra))
            hamScore += hamCounts[palavra];
    }
    Console.WriteLine($"\nPontuação de Spam: {spamScore}");
    Console.WriteLine($"Pontuação de Ham: {hamScore}");
    Console.WriteLine("\n--- RESULTADO DA CLASSIFICAÇÃO ---");
    if (spamScore > hamScore)
        Console.WriteLine("A frase é classificada como: SPAM");
    else if (hamScore > spamScore)
        Console.WriteLine("A frase é classificada como: HAM");
    else
        Console.WriteLine("A frase é classificada como: INDETERMINADA");
}