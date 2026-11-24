# Protocolo de pesquisa (pré-registro) — Brasileirão Predictor

> **Status: RASCUNHO para revisão.** O protocolo só passa a valer quando for commitado (o carimbo do Git é o registro). Depois disso, qualquer mudança vira uma entrada na seção 9, nunca uma edição silenciosa.

## 1. Pergunta de pesquisa
**Um modelo estatístico usando apenas resultados passados consegue prever os resultados do Brasileirão (1X2, gols) tão bem quanto o mercado de apostas?** Se não, quão longe fica, e onde o modelo erra (calibração)?

Não é objetivo deste projeto provar lucro em apostas.

## 2. O que já foi feito é exploratório
Antes de definir hipóteses, é preciso ser honesto sobre o que foi feito até aqui.

| Fato | Consequência |
|---|---|
| As temporadas 2016–2025 foram usadas repetidamente para comparar ~14 variantes de modelo e ~100 configurações de hiperparâmetros (Optuna 2×30 tentativas, grade de 60 do encolhimento). | Os resultados dessas temporadas são **exploratórios**. Os ICs e p-valores não foram corrigidos para comparações múltiplas. |
| A temporada de teste (2025) foi consultada várias vezes, contra a regra do CLAUDE.md ("uma vez por rodada de experimentos"). | 2025 **não é mais um conjunto de teste virgem**. |
| Algumas escolhas (meia-vida 730 dias, ridge 0,5, janela do ρ) foram decididas olhando os dados. | O desempenho de 2016–2025 pode estar levemente otimista. |
| Os achados nulos (GBM, encolhimento, Binomial Negativa, modelo empilhado não superam o Dixon-Coles) foram consistentes entre experimentos. | São o resultado mais robusto, mas ainda exploratório. |

**Regra daqui em diante:** só conta como evidência confirmatória o que for previsto **antes** dos jogos, com modelo e análise congelados (seção 5).

## 3. Hipóteses confirmatórias
Família de 5 hipóteses, testadas no teste forward (seção 5), com correção de Holm (α = 0,05).

| # | Hipótese (H1 é a principal) | Medida | Direção esperada (do exploratório) |
|---|---|---|---|
| **H1** | O RPS do Dixon-Coles difere do RPS das odds de fechamento (média do mercado, sem a margem). | Diferença pareada por jogo, DC − odds | DC **pior** (+0,0076 em 2016–2025) |
| H2 | As probabilidades de mandante ≥ 65% são superestimadas. | Previsto − observado nessa faixa | positivo (+0,04) |
| H3 | O ρ do Dixon-Coles é próximo de zero. | \|ρ\| < 0,05, estimado fora da amostra | confirma (−0,001 nas últimas 6 temporadas) |
| H4 | Não há sobredispersão dos gols dado o λ do modelo. | VMR condicional < 1 | confirma (0,90 e 0,99) |
| H5 | O 1X2 do modelo está bem calibrado. | Erro de calibração (ECE) médio ≤ 0,03, com IC por bootstrap | confirma (0,015) |

Hipótese de **equivalência** (não testável hoje): nenhuma variante com resultados apenas melhora o Dixon-Coles em mais de 0,005 de RPS. Exige n ≈ 1.270 (seção 4); fica como objetivo de longo prazo.

## 4. Poder estatístico: o que dá para concluir com quantos jogos
Dados de 2016–2025: o desvio-padrão da diferença pareada de RPS entre o Dixon-Coles e as odds é **0,0635** por jogo (n = 3.799). Com α = 5% bilateral e poder de 80%:

| Efeito a detectar (diferença de RPS) | Jogos necessários |
|---|---|
| 0,0076 (a lacuna observada para o mercado) | **548** |
| 0,0050 | 1.266 |
| 0,0030 | 3.515 |
| 0,0020 | 7.909 |
| 0,0010 | 31.635 |

| Jogos disponíveis | Menor efeito detectável |
|---|---|
| 21 (o teste forward atual) | **0,0388** |
| 100 | 0,0178 |
| 380 (uma temporada) | 0,0091 |
| 760 (duas temporadas) | 0,0065 |

**Conclusões práticas:**
1. O teste forward de 21 jogos **não consegue** confirmar nem refutar nada sobre a lacuna de 0,0076. Serve só para checar que o pipeline funciona.
2. Para H1 é preciso cerca de 550 jogos: a temporada 2027 inteira (380) mais 2026 restante (21) dá 401, ou seja, cerca de 65% de poder. Com 2028, passa de 780.
3. **O desvio da diferença pareada depende de quão parecidos são os modelos**, e a tabela acima vale só para comparações com o mercado ou com modelos muito diferentes (DC contra odds, Elo ou GBM: desvio de 0,05 a 0,06). Para um ajuste aninhado, como somar uma variável ao Dixon-Coles (DC contra DC + valor do time titular), o desvio medido foi **0,0114**: bastam cerca de 115 jogos para detectar 0,003 e cerca de 1.600 para detectar 0,0008. *(Correção: a versão anterior deste item dizia que melhorias abaixo de 0,005 não eram resolvíveis com uma liga; isso só vale para modelos muito diferentes.)* Ainda assim, um efeito pequeno detectável não é um efeito relevante: ganhos de 0,001 de RPS não aproximam o modelo do mercado (lacuna de 0,0076).
4. Alternativa para aumentar n: aplicar o mesmo protocolo a outras ligas que o football-data.co.uk cobre (exige decisão sua; mudaria o escopo).

## 5. Plano do teste forward
- **Modelo congelado:** Dixon-Coles de produção (`MODEL_TAG` em `src/forward.py`), meia-vida 730 dias, ridge 0,5, ρ fora da amostra das últimas 6 temporadas. **Código:** commit `b73901e`. **Dados de treino:** `BRA_footballdata.csv`, SHA-256 `e3d47bdc6d741104387adddbb52c4c6d00064816b4647ae5958f1bfaa94ccb7b` (24/11/2026).
- **Registro:** `python -m src.forward log` antes de cada rodada; arquivo imutável em `predictions/`, commitado antes dos jogos. Previsões de rodadas futuras podem ser registradas a cada atualização (`python -m src.update`).
- **Comparador:** odds de fechamento médias do mercado (`AvgC*`), sem a margem (normalização proporcional), nos mesmos jogos.
- **Métrica primária:** RPS. Secundárias: log-loss, Brier, calibração, Over 2.5 (Brier).
- **Análise:** diferença pareada por jogo; IC 95% por bootstrap pareado (2.000 repetições, seed 42); correção de Holm sobre H1–H5. Análise final **só** quando n ≥ 548 jogos registrados ou ao fim da temporada 2027, o que vier depois. Sem análises intermediárias para decidir parada.
- **Inclusão:** todo jogo previsto e disputado entra, sem exceções (não escolher jogos).

## 6. Regras contra p-hacking e deriva
1. **Uma versão de modelo por vez no forward.** Trocar modelo, hiperparâmetro ou janela durante o teste só vale como **nova versão** (`MODEL_TAG` novo), avaliada em separado e sem substituir a original.
2. Variantes novas precisam de hipótese e efeito esperado **antes** de rodar, com o tamanho de amostra necessário calculado a partir do desvio da diferença pareada esperado (seção 4, item 3).
3. Todo experimento exploratório continua permitido, mas é registrado no CLAUDE.md como exploratório, com o número de comparações feitas.
4. Os resultados de 2016–2025 não são reutilizados como evidência confirmatória.
5. Resultados nulos e negativos são publicados da mesma forma que os positivos.

## 7. Reprodutibilidade
- Sementes fixas (`SEED = 42`); ordem das features salva com o modelo.
- Cada log do forward guarda: data da previsão, modelo, treino até quando. **A fazer:** guardar também o hash do CSV e o commit do código (hoje só aparecem neste protocolo).
- Versões das bibliotecas em `requirements.txt`; **a fazer:** registrar as versões instaladas (`pip freeze`) junto ao protocolo.
- Todo resultado reproduzível por comando (`src.backtest`, `src.eda`, `src.forward`, ...).
- Dados brutos imutáveis (verificado por teste); conferência de autenticidade em `reports/crosscheck.md` (1.140 de 1.140 jogos de 2022–2024 batem com a API-Football).

## 8. Ameaças à validade
| Tipo | Ameaça | Mitigação |
|---|---|---|
| Interna | Vazamento de dados | Testes de invariância por truncamento e por mutação; λs de treino fora da amostra |
| Interna | Seleção de modelo olhando o teste | Seção 2; protocolo forward |
| Conclusão | Comparações múltiplas e baixo poder | Holm; tabelas de poder; não concluir com n pequeno |
| Externa | Uma só liga e um só país | Declarar o limite; opcionalmente replicar em outras ligas |
| Externa | Regime muda (calendário, regras, elenco) | Decaimento temporal; monitorar a calibração por temporada |
| Construto | Odds de fechamento já incorporam lesões e escalações | Comparar também só com odds de abertura se houver fonte; documentar a diferença |
| Construto | RPS pode ocultar erros em subconjuntos (favoritos, empates) | Calibração por faixa de probabilidade |

## 9. Registro de desvios
| Data | Desvio | Motivo | Impacto na análise |
|---|---|---|---|
| — | (nenhum até agora) | — | — |

## 10. Próximos passos propostos
1. Revisar este rascunho e fechar H1–H5 (cortar, trocar ou acrescentar).
2. Decidir se o escopo inclui outras ligas para ganhar poder (seção 4, item 4).
3. Implementar o registro do hash do CSV e do commit em `src/forward.py` (seção 7).
4. Commitar o protocolo e só então registrar novas previsões.
