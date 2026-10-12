"""
Télécharge les images des peintures depuis les pages wga.hu référencées dans
data/paintings_only.csv (colonne URL).

Mapping observé sur le site : une page HTML
    https://www.wga.hu/html/{a}/{artiste}/{oeuvre}.html
a toujours son image pleine résolution à
    https://www.wga.hu/art/{a}/{artiste}/{oeuvre}.jpg

"""


import argparse
import csv
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

USER_AGENT = "Mozilla/5.0 (compatible; EDA-research-script/1.0; personal/educational GAN dataset project)"


def html_url_to_image_url(html_url: str) -> str:
    return html_url.replace("/html/", "/art/").replace(".html", ".jpg")


def target_path(html_url: str, out_dir: Path) -> Path:
    # mirrors the /a/artiste/oeuvre.jpg structure sous out_dir/, pour garantir
    # des noms de fichiers uniques et pouvoir retrouver la ligne CSV d'origine.
    tail = html_url.split("/html/", 1)[1]  # ex: a/aachen/adonis.html
    return out_dir / Path(tail).with_suffix(".jpg")


def download_one(session, html_url, out_dir, timeout, min_delay, max_retries) -> tuple:
    img_url = html_url_to_image_url(html_url)
    dest = target_path(html_url, out_dir)

    if dest.exists() and dest.stat().st_size > 0:
        return html_url, "skip (déjà présent)"

    dest.parent.mkdir(parents=True, exist_ok=True)

    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(min_delay + random.uniform(0, 0.2))
            resp = session.get(img_url, timeout=timeout)
            if resp.status_code == 200 and resp.content:
                dest.write_bytes(resp.content)
                return html_url, "ok"
            last_err = f"http {resp.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(1.5 * attempt)  # backoff
    return html_url, f"échec ({last_err})"


def scrape(csv_path, out_dir, limit=None, workers=4, min_delay=0.4, timeout=20, max_retries=3):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_log = out_dir / "_failed.csv"

    with open(csv_path, encoding="utf-8") as f:
        urls = [row["URL"] for row in csv.DictReader(f)]
    if limit:
        urls = urls[:limit]

    print(f"{len(urls):,} images à traiter (concurrence = {workers})")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    ok = skip = fail = 0
    failures = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(download_one, session, u, out_dir, timeout, min_delay, max_retries): u
            for u in urls
        }
        for i, fut in enumerate(as_completed(futures), 1):
            url, status = fut.result()
            if status == "ok":
                ok += 1
            elif status.startswith("skip"):
                skip += 1
            else:
                fail += 1
                failures.append((url, status))

            if i % 200 == 0 or i == len(urls):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  {i:,}/{len(urls):,}  ok={ok} skip={skip} fail={fail}  "
                      f"({rate:.1f} img/s, {elapsed/60:.1f} min écoulées)")

    if failures:
        with open(failed_log, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["URL", "erreur"])
            w.writerows(failures)
        print(f"\n{len(failures)} échecs loggés dans {failed_log}")

    print(f"\nTerminé en {(time.time()-t0)/60:.1f} min — ok={ok} skip={skip} fail={fail}")
    return ok, skip, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/paintings_only.csv", help="CSV source (colonne URL)")
    ap.add_argument("--out", default="images", help="dossier de destination des images")
    ap.add_argument("--limit", type=int, default=None, help="ne traiter que les N premières lignes (test)")
    ap.add_argument("--workers", type=int, default=4, help="requêtes en parallèle")
    ap.add_argument("--min-delay", type=float, default=0.4, help="délai mini entre deux requêtes par worker (s)")
    ap.add_argument("--timeout", type=float, default=20, help="timeout réseau par requête (s)")
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    scrape(args.csv, args.out, args.limit, args.workers, args.min_delay, args.timeout, args.max_retries)


if __name__ == "__main__":
    main()
