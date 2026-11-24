# Brasileirão Predictor

Previsão de placares e probabilidades (1X2, Over 2.5, BTTS) do Campeonato Brasileiro Série A, usando só informação disponível **antes do jogo**.

O modelo de produção é um **Dixon-Coles** com decaimento temporal (Poisson para gols, correção de placares baixos). O valor do projeto está menos no algoritmo, que é clássico, e mais na avaliação: tudo é medido contra as odds de fechamento das casas de apostas, com testes contra vazamento de dados e um teste forward registrado no Git.

> **Resumo honesto:** o modelo fica cerca de 0,007 de RPS atrás das odds de fechamento. Não há evidência de vantagem para apostas.

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

Experimentos (todos reproduzíveis):

| Experimento | Comando |
|---|---|
| Baselines e backtest walk-forward | `python -m src.backtest` |
| LightGBM Poisson com Optuna | `python -m src.train` |
| GBM sobre o Dixon-Coles (offset) | `python -m src.stacking` |
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
| LightGBM Poisson (53 features) | 0,2107 | 1,030 |
| Frequências históricas | 0,2188 | 1,054 |

O que testamos e **não** melhorou o Dixon-Coles (intervalos de confiança incluem zero ou pioram):
- LightGBM com 53 features de forma, descanso, H2H e Elo.
- LightGBM usando o Dixon-Coles como base.
- Encolhimento dos λs (ridge, meia-vida, α).
- Binomial Negativa: os gols não têm sobredispersão (a dispersão ajustada é ~0), e o ρ saiu positivo (+0,01 a +0,02), ao contrário das ligas europeias.

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
                 stacking, shrinkage, dispersion, calibration, predict, forward, update
tests/           60 testes (vazamento, matriz soma 1, ingestão, inferência, update)
CLAUDE.md        especificação do projeto, decisões e registro de resultados
```

Modelos treinados (`models/`) e arquivos gerados (`data/interim/`, `reports/`) não vão para o Git; `python -m src.update` os recria.

## Aviso

Projeto de estudo. As probabilidades não são conselho financeiro nem recomendação de aposta.
