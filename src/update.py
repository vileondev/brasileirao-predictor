"""One-shot refresh: download results, validate, refit, score the forward test, log new fixtures.

    python -m src.update              # full refresh
    python -m src.update --no-download  # reuse the local CSV
    python -m src.update --no-log       # do not freeze new predictions

The downloaded CSV replaces ``data/raw/BRA_footballdata.csv`` only if it passes
schema validation; otherwise the current file is kept untouched.
"""
from __future__ import annotations

import argparse
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from src import config, forward, ingestion, leagues, predict


def download_raw(url: str = config.RAW_URL, dest: Path = config.RAW_MATCHES_FILE) -> int:
    """Download the CSV, validate it and replace ``dest`` atomically.

    Returns
    -------
    int
        Number of newly available played matches compared with the previous file
        (negative if the new file has fewer rows).

    Raises
    ------
    ingestion.SchemaError
        If the downloaded file fails validation (``dest`` is left untouched).
    RuntimeError
        On network problems.
    """
    tmp = dest.with_suffix(".download")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "brasileirao-predictor/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
    except (urllib.error.URLError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {e}") from None
    try:
        new = ingestion.build_interim(tmp)
    except ingestion.SchemaError:
        tmp.unlink(missing_ok=True)
        raise
    old_n = len(ingestion.build_interim(dest)) if dest.exists() else 0
    tmp.replace(dest)
    return len(new) - old_n


def new_matchups(log: pd.DataFrame, directory: Path = forward.PRED_DIR) -> pd.DataFrame:
    """Keep only matchups that were not already frozen in an earlier log."""
    done = forward.load_logs(directory)
    if done.empty:
        return log
    seen = set(zip(done["season"], done["home"], done["away"]))
    mask = [(s, h, a) not in seen for s, h, a in zip(log["season"], log["home"], log["away"])]
    return log[mask].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--league", default="BRA", help="league code: " + ", ".join(sorted(leagues.LEAGUES)))
    args = ap.parse_args()
    lg = leagues.get(args.league)

    if not args.no_download:
        try:
            if lg.source == "football-data-new":
                added = download_raw()
                print(f"[1/4] Dados atualizados: {added:+d} jogos disputados desde a ultima versao.")
            else:
                res = leagues.download_football_data(lg)
                print(f"[1/4] {lg.code}: {len(res['changed'])} arquivo(s) novo(s)/atualizado(s): {', '.join(res['changed']) or '—'}")
        except ingestion.SchemaError as e:
            print(f"[1/4] Arquivo baixado INVALIDO, mantive o atual: {e}\n"
                  "      (se for time novo, acrescente o nome em TEAM_MAP em src/config.py)")
            return
        except RuntimeError as e:
            print(f"[1/4] Sem download ({e}); usando o arquivo local.")
    else:
        print("[1/4] Download pulado.")

    matches = ingestion.build_league(lg)
    last = matches["kickoff"].max()
    print(f"      [{lg.code}] {len(matches)} jogos, ultimo em {last:%d/%m/%Y}.")

    art = predict.fit_production(matches)
    predict.save_artifact(art, lg.model_file)
    print(f"[2/4] Modelo reajustado (rho={art['rho']:.3f}) e salvo em {lg.model_file.relative_to(config.ROOT)}.")

    logs = forward.load_logs(lg.pred_dir)
    print("[3/4] Forward test:")
    if logs.empty:
        print("      nenhuma previsao registrada ainda.")
    else:
        forward._report(forward.score_log(logs, matches), len(logs))

    if args.no_log:
        print("[4/4] Registro de novas previsoes pulado.")
        return
    new = new_matchups(forward.make_log(matches, league=lg), lg.pred_dir)
    if new.empty:
        print("[4/4] Nenhum confronto novo para registrar.")
        return
    try:
        path = forward.save_log(new, lg.pred_dir)
    except FileExistsError:
        alt = new.copy()
        alt["predicted_at"] = str(pd.Timestamp.now().floor("s"))
        path = lg.pred_dir / f"{str(alt['predicted_at'].iloc[0])[:10]}_{len(new)}new_remaining.csv"
        alt.to_csv(path, index=False)
    print(f"[4/4] {len(new)} novas previsoes registradas em {path.name} (faca commit para fixar o carimbo).")


if __name__ == "__main__":
    main()
