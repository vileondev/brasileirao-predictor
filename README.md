# Brasileirão Predictor

Previsão de placares e probabilidades (1X2, Over 2.5, BTTS) do Campeonato Brasileiro Série A, usando só informação disponível **antes do jogo**.

O modelo de produção é um **Dixon-Coles** com decaimento temporal (Poisson para gols, correção de placares baixos). O valor do projeto está menos no algoritmo, que é clássico, e mais na avaliação: tudo é medido contra as odds de fechamento das casas de apostas, com testes contra vazamento de dados e um teste forward registrado no Git.

> **Resumo honesto:** o modelo fica cerca de 0,007 de RPS atrás das odds de fechamento. Não há evidência de vantagem para apostas.

## Probabilidades, não odds

O modelo não prevê as odds das casas de apostas: ele prevê **probabilidades** (a odd justa é 1 dividido pela probabilidade). Uma única matriz de placares dá o 1X2, o Over 2.5, o BTTS e os placares mais prováveis, e permite medir a qualidade com RPS, log-loss e calibração. As odds de fechamento entram só como régua de comparação. Como o modelo é pior que o mercado, uma probabilidade muito diferente da odd de mercado deve ser lida como alerta de informação que falta (lesão, escalação reserva), e não como oportunidade de aposta.

## Instalação

```powershell
pip install -r requirements.txt
python -m pytest tests -q
```

Requer Python 3.11+ (desenvolvido em 3.13). Os dados brutos já estão em `data/raw/`.

## Uso

| Objetivo | Comando |
|---|---|
| Atualizar dados, reajustar o modelo, pontuar o forward test e registrar novas previsões | `python -m src.update` |
| Prever um jogo | `python -m src.predict --refit --match Flamengo Palmeiras` |
| Prever os jogos que faltam na temporada (sem datas) | `python -m src.predict --remaining` |
| Prever uma lista (CSV com colunas `home,away`) | `python -m src.predict --fixtures fixtures.csv` |
| Congelar previsões antes dos jogos | `python -m src.forward log` |
| Pontuar as previsões congeladas | `python -m src.forward score` |

Exemplo de saída:

```text
Flamengo x Palmeiras   (xG 1.46 - 1.06)
  1X2: 46.4% / 25.9% / 27.7%   Over2.5: 46.3%   BTTS: 50.2%
  Placares: 1-1 (12.3%) | 1-0 (11.8%) | 2-1 (9.1%)
```

## Como ler a saída

Usando o exemplo acima:

```text
Flamengo x Palmeiras   (xG 1.46 - 1.06)
  1X2: 46.4% / 25.9% / 27.7%   Over2.5: 46.3%   BTTS: 50.2%
  Placares: 1-1 (12.3%) | 1-0 (11.8%) | 2-1 (9.1%)
```

| Trecho | O que significa |
|---|---|
| `Flamengo x Palmeiras` | O primeiro time é o **mandante**. Inverter a ordem muda o resultado, porque jogar em casa aumenta os gols esperados em cerca de 40% (na média da história, 1,43 gol do mandante contra 0,98 do visitante). |
| `xG 1.46 - 1.06` | Gols esperados de cada time (mandante - visitante). **Não é o xG de chutes**: é a média de gols que o modelo espera, calculada só com resultados passados. |
| `1X2: 46.4% / 25.9% / 27.7%` | Probabilidade de vitória do mandante, empate e vitória do visitante, nessa ordem. Somam 100%. |
| `Over2.5: 46.3%` | Probabilidade de saírem 3 gols ou mais no jogo. |
| `BTTS: 50.2%` | Probabilidade de os dois times marcarem. |
| `Placares: 1-1 (12.3%) \| ...` | Os 3 placares exatos mais prováveis, com a probabilidade de cada um (mandante primeiro). |

Pontos para não se enganar:
- **O placar mais provável quase nunca é o mais provável de acontecer.** No exemplo, o 1-1 tem 12,3%, ou seja, a chance de *não* sair 1-1 é de 87,7%. Os 3 placares juntos somam só 33,2%, e o resto se espalha por dezenas de outros placares. Por isso o foco do projeto é nas probabilidades, e não em acertar o placar exato.
- **O favorito não é o placar.** Aqui o Flamengo é o favorito (46,4%), mas o placar mais provável é um empate. As duas leituras são coerentes: o empate é um único placar, e a vitória do mandante reúne vários (1-0, 2-1, 2-0...).
- **Probabilidade vira odd justa** dividindo 1 por ela: 46,4% corresponde a uma odd de cerca de 2,16, o empate a 3,86 e o visitante a 3,61. Odds de casas de apostas têm margem e informação que o modelo não tem, então não são comparáveis diretamente.
- **O modelo não conhece escalação nem lesões.** Trate a saída como uma estimativa baseada na força recente dos times, não como certeza.

## Experimentos

Todos reproduzíveis:

| Experimento | Comando |
|---|---|
| Baselines e backtest walk-forward | `python -m src.backtest` |
| LightGBM Poisson com Optuna | `python -m src.train` |
| GBM sobre o Dixon-Coles (offset) | `python -m src.stacking` |
| Modelo empilhado time contra adversário | `python -m src.stacked` |
| Encolhimento dos λs | `python -m src.shrinkage` |
| Poisson contra Binomial Negativa | `python -m src.dispersion` |
| Calibração (1X2 e Over 2.5) | `python -m src.calibration` |

## Dados

- **Fonte:** [football-data.co.uk](https://www.football-data.co.uk/brazil.php) (`BRA.csv`): Série A de 2012 a 2026, cerca de 5.700 jogos, com resultados e odds de fechamento. Não tem xG, chutes nem escalações.
- Um jogo nunca disputado (Chapecoense x Atlético-MG, 2016, após o acidente aéreo) é descartado na ingestão.
- A API-Football (plano grátis) só libera as temporadas 2022–2024, então não é usada para jogos futuros. A chave, se usada, fica em um `.env` fora do Git (`API_FOOTBALL_KEY`).

## Resultados (walk-forward 2016–2025, RPS: menor é melhor)

| Modelo | RPS | Log-loss |
|---|---|---|
| Odds de fechamento (Pinnacle) | 0,2002 | 0,997 |
| Odds de fechamento (média do mercado) | 0,2005 | 0,998 |
| **Dixon-Coles (produção)** | **0,2081** | 1,022 |
| Elo + Poisson | 0,2082 | 1,024 |
| GBM com offset do Dixon-Coles | 0,2080 | 1,022 |
| Empilhado time x adversário, com offset do Dixon-Coles | 0,2083 | 1,023 |
| Empilhado time x adversário, puro | 0,2101 | 1,029 |
| LightGBM Poisson (53 features) | 0,2107 | 1,030 |
| Frequências históricas | 0,2188 | 1,054 |

O que testamos e **não** melhorou o Dixon-Coles (intervalos de confiança incluem zero ou pioram):
- LightGBM com 53 features de forma, descanso, H2H e Elo.
- LightGBM usando o Dixon-Coles como base.
- Modelo único empilhado (duas linhas por jogo, uma por time): dobrar as linhas de treino não ajuda.
- Encolhimento dos λs (ridge, meia-vida, α).
- Binomial Negativa: os gols não têm sobredispersão (a dispersão ajustada é ~0).

Sobre o ρ do Dixon-Coles: ele é pequeno e depende da janela usada (de +0,03 a +0,10 nas primeiras temporadas, −0,001 nas últimas 6, +0,011 no ajuste dentro da amostra). Nunca chega aos valores negativos das ligas europeias, ou seja, não há excesso de empates de placar baixo. A produção usa a estimativa fora da amostra das últimas 6 temporadas.

Calibração: o 1X2 está bem calibrado (erro médio 0,015, contra 0,013 das odds). O Over 2.5 é um pouco otimista nas probabilidades altas.

## Diferenciais e limites

**O que diferencia o projeto**
- Benchmark contra o mercado, com bootstrap pareado.
- Resultados negativos registrados, não escondidos.
- Anti-vazamento testado: invariância por truncamento do futuro e por mutação do próprio resultado, em features, Elo e backtest; λs de treino fora da amostra.
- Forward test: previsões congeladas em `predictions/` e commitadas antes dos jogos.

**O que não é diferencial**
- O modelo de produção é o Dixon-Coles de 1997.
- Não superamos as odds de fechamento.
- Sem dados de jogador (escalações, lesões, valor de elenco): é a maior lacuna. O Transfermarkt bloqueia robôs de IA no `robots.txt` e não é usado.
- O forward test atual tem só 21 jogos (rodadas 37 e 38 de 2026) e não prova nada ainda.
- Uma única fonte de dados, sem xG.

## Estrutura

```text
data/raw/        dados brutos imutáveis
predictions/     previsões congeladas (forward test)
src/             ingestion, features, baselines, probability, backtest, train,
                 stacking, stacked, shrinkage, dispersion, calibration, predict, fixtures,
                 forward, update
tests/           64 testes (vazamento, matriz soma 1, ingestão, inferência, update, forward)
CLAUDE.md        especificação do projeto, decisões e registro de resultados
```

Modelos treinados (`models/`) e arquivos gerados (`data/interim/`, `reports/`) não vão para o Git; `python -m src.update` os recria.

## Aviso

Projeto de estudo. As probabilidades não são conselho financeiro nem recomendação de aposta.
