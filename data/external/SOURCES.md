# Fontes dos dados externos

Estes arquivos foram baixados para pesquisa e **publicados por decisão do dono do repositório em 25/11/2026, depois de uma revisão dele**. Os termos de uso de cada fonte **não foram verificados por quem montou o pipeline** (veja o TODO do README). Se algum detentor de direitos pedir, os arquivos devem ser removidos.

| Pasta | Origem | Baixado em | Observações |
|---|---|---|---|
| `williamorim_brasileirao/` (`stats.csv`, `matches.csv`) | [github.com/williamorim/brasileirao](https://github.com/williamorim/brasileirao) (código sob licença MIT, © 2021 William Nilson de Amorim) | 24/11/2026 | `matches.csv` vem de scraping do chancedegol (2003–2019) e do Globo Esporte (2020+). `stats.csv` vem de uma tabela da Base dos Dados, provavelmente derivada do Transfermarkt (**não confirmado**). Cobre 2003–2020. SHA-256 em `CLAUDE.md`. |
| `football_data_england/` (`E0_*.csv`, `fixtures.csv`) | [football-data.co.uk](https://football-data.co.uk) | 24/11/2026 | Premier League 2012/13–2026/27 e calendário das ligas europeias. Hashes em `MANIFEST.json`. |
| `api_football/` | [API-Football](https://www.api-football.com) (plano grátis, chave própria) | 24–25/11/2026 | Listas de jogos 2022–2024, lesões (Premier League 2022–2024, Brasileirão 2024) e estatísticas por jogo (xG, chutes, posse). O download do xG continua aos poucos: `python -m src.xg_api download`. |

## Como cada pasta é usada
- `williamorim_brasileirao/`: conferência de placares e testes exploratórios do valor do time titular (`src/eda_external.py`).
- `football_data_england/`: baselines da Premier League (exploratório).
- `api_football/`: armazenamento retomável de xG e lesões (`src/xg_api.py`). Os arquivos de `stats/` crescem a cada execução; commitar de tempos em tempos.

Nada aqui é conselho financeiro nem recomendação de aposta.
