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
| Prever um jogo | `python -m src.predict --refit --match Palmeiras Flamengo` |
| Prever os jogos que faltam na temporada, com data e hora | `python -m src.predict --remaining` |
| Prever uma lista (CSV com colunas `home,away`) | `python -m src.predict --fixtures fixtures.csv` |
| Congelar previsões antes dos jogos | `python -m src.forward log` |
| Pontuar as previsões congeladas | `python -m src.forward score` |

Exemplo de saída:

```text
Palmeiras x Flamengo   (xG 1.49 - 1.04)
  1X2: 47.8% / 25.7% / 26.5%   Over2.5: 46.5%   BTTS: 50.1%
  Placares: 1-1 (12.2%) | 1-0 (12.0%) | 2-1 (9.2%)
```

## Como ler a saída

Usando o exemplo acima: Palmeiras x Flamengo, de 21/11/2026, que terminou **1 x 1**. A previsão é a do modelo atual, que já tem esse resultado no treino; serve só para ilustrar o formato.

```text
Palmeiras x Flamengo   (xG 1.49 - 1.04)
  1X2: 47.8% / 25.7% / 26.5%   Over2.5: 46.5%   BTTS: 50.1%
  Placares: 1-1 (12.2%) | 1-0 (12.0%) | 2-1 (9.2%)
```

| Trecho | O que significa |
|---|---|
| `Palmeiras x Flamengo` | O primeiro time é o **mandante**. Inverter a ordem muda o resultado, porque jogar em casa aumenta os gols esperados em cerca de 40% (na média da história, 1,43 gol do mandante contra 0,98 do visitante). |
| `xG 1.49 - 1.04` | Gols esperados de cada time (mandante - visitante). **Não é o xG de chutes**: é a média de gols que o modelo espera, calculada só com resultados passados. |
| `1X2: 47.8% / 25.7% / 26.5%` | Probabilidade de vitória do mandante (Palmeiras), empate e vitória do visitante (Flamengo), nessa ordem. Somam 100%. |
| `Over2.5: 46.5%` | Probabilidade de saírem 3 gols ou mais no jogo. |
| `BTTS: 50.1%` | Probabilidade de os dois times marcarem. |
| `Placares: 1-1 (12.2%) \| ...` | Os 3 placares exatos mais prováveis, com a probabilidade de cada um (mandante primeiro). |

Pontos para não se enganar:
- **O placar mais provável quase nunca é o mais provável de acontecer.** No exemplo, o 1-1 tem 12,2%, ou seja, a chance de *não* sair 1-1 é de 87,8%. Os 3 placares juntos somam só 33,4%, e o resto se espalha por dezenas de outros placares. Por isso o foco do projeto é nas probabilidades, e não em acertar o placar exato.
- **O favorito não é o placar.** Aqui o Palmeiras é o favorito (47,8%), mas o placar mais provável é um empate (1-1, justamente o que aconteceu). As duas leituras são coerentes: o empate é um único placar, e a vitória do mandante reúne vários (1-0, 2-1, 2-0...).
- **Probabilidade vira odd justa** dividindo 1 por ela: 47,8% corresponde a uma odd de cerca de 2,09, o empate a 3,89 e o visitante a 3,77. Odds de casas de apostas têm margem e informação que o modelo não tem, então não são comparáveis diretamente.
- **O modelo não conhece escalação nem lesões.** Trate a saída como uma estimativa baseada na força recente dos times, não como certeza.

## Próximos jogos e previsões

Os **21 jogos que faltam** na Série A 2026, em ordem cronológica. As previsões foram congeladas em 24/11/2026, antes das partidas, e são as do forward test (`predictions/2026-11-24_remaining.csv`). O calendário foi digitado à mão a partir de uma página pública de tabela (`data/raw/schedule_2026_manual.csv`); os horários são locais.

| Quando | Jogo (mandante x visitante) | Mandante / Empate / Visitante | Gols esperados | Placar mais provável | Over 2.5 | BTTS |
|---|---|---|---|---|---|---|
| Ter, 24/11, 18h (rod. 37) | Chapecoense x Palmeiras | 12% / 19% / **68%** | 0,7–2,1 | 0-2 | 53% | 45% |
| Qua, 25/11, 20h (rod. 36) | Atlético-MG x Corinthians | 48% / 27% / 25% | 1,4–0,9 | 1-0 | 40% | 45% |
| Qui, 26/11, 20h (rod. 37) | São Paulo x Remo | **60%** / 23% / 17% | 1,8–0,9 | 1-0 | 49% | 48% |
| Sex, 27/11, 20h (rod. 37) | Fluminense x Cruzeiro | 44% / 28% / 28% | 1,3–1,0 | 1-0 | 40% | 46% |
| Sáb, 28/11, 21h (rod. 37) | Botafogo x Bahia | 51% / 25% / 24% | 1,6–1,0 | 1-1 | 49% | 52% |
| Dom, 29/11, 16h (rod. 37) | Coritiba x Flamengo | 19% / 22% / **59%** | 1,0–1,9 | 0-1 | 54% | 52% |
| Dom, 29/11, 16h (rod. 37) | Vitória x Santos | 41% / 26% / 32% | 1,4–1,2 | 1-1 | 47% | 51% |
| Dom, 29/11, 18h30 (rod. 37) | Corinthians x Grêmio | 48% / 26% / 26% | 1,4–1,0 | 1-0 | 43% | 47% |
| Dom, 29/11, 18h30 (rod. 37) | Internacional x Bragantino | 48% / 26% / 26% | 1,5–1,0 | 1-0 | 45% | 49% |
| Dom, 29/11, 18h30 (rod. 37) | Mirassol x Athletico-PR | 48% / 26% / 26% | 1,5–1,0 | 1-1 | 47% | 50% |
| Dom, 29/11, 19h30 (rod. 37) | Atlético-MG x Vasco | **56%** / 23% / 21% | 1,7–1,0 | 1-0 | 50% | 50% |
| Qua, 02/12, 21h30 (rod. 38) | Athletico-PR x São Paulo | 47% / 27% / 26% | 1,4–1,0 | 1-0 | 41% | 46% |
| Qua, 02/12, 21h30 (rod. 38) | Bahia x Atlético-MG | 43% / 27% / 30% | 1,4–1,1 | 1-1 | 45% | 50% |
| Qua, 02/12, 21h30 (rod. 38) | Bragantino x Fluminense | 41% / 27% / 32% | 1,3–1,1 | 1-1 | 43% | 48% |
| Qua, 02/12, 21h30 (rod. 38) | Cruzeiro x Internacional | 52% / 26% / 22% | 1,5–0,9 | 1-0 | 43% | 46% |
| Qua, 02/12, 21h30 (rod. 38) | Flamengo x Chapecoense | **87%** / 9% / 4% | 3,2–0,6 | 3-0 | 72% | 42% |
| Qua, 02/12, 21h30 (rod. 38) | Grêmio x Mirassol | 43% / 26% / 30% | 1,4–1,1 | 1-1 | 46% | 51% |
| Qua, 02/12, 21h30 (rod. 38) | Palmeiras x Coritiba | **77%** / 15% / 8% | 2,4–0,6 | 2-0 | 58% | 42% |
| Qua, 02/12, 21h30 (rod. 38) | Remo x Corinthians | 35% / 26% / 39% | 1,2–1,3 | 1-1 | 47% | 51% |
| Qua, 02/12, 21h30 (rod. 38) | Santos x Botafogo | 42% / 26% / 32% | 1,4–1,2 | 1-1 | 47% | 52% |
| Qua, 02/12, 21h30 (rod. 38) | Vasco x Vitória | 51% / 24% / 24% | 1,7–1,1 | 1-1 | 51% | 53% |

Em negrito, as probabilidades de 55% ou mais. O placar mais provável tem só 11% a 14% de chance, então olhe as probabilidades de resultado antes do placar. Esta tabela é uma foto: o resultado de cada jogo entra quando o football-data.co.uk atualizar, e `python -m src.update` mostra como as previsões se saíram.

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
- O calendário dos jogos restantes de 2026 está em `data/raw/schedule_2026_manual.csv`, digitado à mão a partir de uma página pública (horário local). Quando a temporada acabar, ele deixa de ser necessário; para 2027 é preciso preencher um novo.
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
tests/           66 testes (vazamento, matriz soma 1, ingestão, inferência, update, forward)
CLAUDE.md        especificação do projeto, decisões e registro de resultados
```

Modelos treinados (`models/`) e arquivos gerados (`data/interim/`, `reports/`) não vão para o Git; `python -m src.update` os recria.

## TODO: o que você precisa correr atrás

Itens que dependem de **mim** (decisão, conta, dado ou espera).

### Já, esta semana
- [x] **Enviar os commits ao GitHub** (feito em 24/11/2026; lembrar de repetir o `git push` a cada novo commit, principalmente os das previsões congeladas).
- [ ] **Depois dos jogos das rodadas 36–38** (24/11 a 02/12/2026): baixar o CSV novo do football-data.co.uk e rodar `python -m src.update`. Ele pontua as 21 previsões congeladas. *Com 21 jogos é só acompanhamento, não conclui nada.*
- [ ] **Revisar o protocolo** (`docs/PROTOCOLO.md`): cortar, trocar ou acrescentar hipóteses (H1 a H5), tirar o aviso de "RASCUNHO" e commitar. Só vale como pré-registro depois do commit.

### Decisões
- [ ] **Escopo do protocolo:** continuar só com o Brasileirão (precisa de 2027 inteiro, cerca de 400 jogos, e ainda com poder de cerca de 65%) ou aplicar o mesmo protocolo a outras ligas do football-data.co.uk para ganhar poder mais rápido.
- [ ] **Dados de jogador:** decidir se vale insistir. O que sobrou é uma fonte gratuita **permitida** de lesões e escalações da temporada atual (por exemplo, as súmulas da CBF: conferir os termos antes). O Transfermarkt e o FBref estão fora (bloqueiam robôs de IA), e a API-Football grátis não libera 2025 e 2026.

### Verificações de termos de uso (só eu posso fazer)
- [ ] **Base dos Dados** (basedosdados.org): confirmar fonte e licença da tabela `stats.csv`. Hoje ela fica fora do Git (`data/external/`) por isso.
- [ ] **Globo Esporte:** a tabela `matches.csv` do `williamorim/brasileirao` vem de scraping dele; os termos de uso não foram verificados.
- [ ] **football-data.co.uk:** conferir se pode redistribuir o CSV (`data/raw/`) num repositório público.

### Antes da temporada 2027
- [ ] **Calendário:** preencher um novo `data/raw/schedule_*_manual.csv` (ou confirmar que o scraper do `williamorim/brasileirao` ganhou a função de 2027).
- [ ] **Times promovidos:** acrescentar os nomes novos em `TEAM_MAP` (`src/config.py`); o `python -m src.update` avisa se faltar algum.
- [ ] **Rodar `python -m src.update` antes de cada rodada**, para congelar as previsões e commitá-las antes dos jogos (o carimbo do commit é a prova).
- [ ] **Análise final do protocolo** só com pelo menos 548 jogos registrados (ou ao fim da temporada 2027).

> Nota: se a API-Football falhar com erro de DNS, a causa é o DNS do provedor; trocar para `1.1.1.1` e `8.8.8.8` resolve.


## Aviso

Projeto de estudo. As probabilidades não são conselho financeiro nem recomendação de aposta.
