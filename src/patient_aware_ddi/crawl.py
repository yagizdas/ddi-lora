"""Extract DDInter interaction records with robots checks and request pacing."""

import argparse
import time
import random
import re
import csv
import json
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests
import pandas as pd
from bs4 import BeautifulSoup


BASE_URL = "https://ddinter.scbdd.com"
INTERACTION_URL_TEMPLATE = BASE_URL + "/ddinter/interact/{interaction_id}/"

OUTPUT_CSV = "data/raw/ddinter_interactions.csv"
OUTPUT_JSONL = "data/raw/ddinter_interactions.jsonl"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-CH-UA": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
    "Cache-Control": "max-age=0",
}


def allowed_by_robots(url: str, user_agent: str = HEADERS["User-Agent"]) -> bool:
    """
    Checks whether robots.txt allows fetching the given URL.
    """
    robots_url = urljoin(BASE_URL, "/robots.txt")
    rp = RobotFileParser()
    rp.set_url(robots_url)

    try:
        rp.read()
    except Exception:
        return False

    return rp.can_fetch(user_agent, url)


def fetch_page(session: requests.Session, url: str, timeout: int = 20) -> str | None:
    if not allowed_by_robots(url):
        print(f"[ROBOTS BLOCKED] {url}")
        return None

    try:
        response = session.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException as exc:
        print(f"[REQUEST ERROR] {url}: {exc}")
        return None

    if response.status_code == 200:
        return response.text

    if response.status_code in {403, 429}:
        print(f"[STOP] Received {response.status_code} for {url}. Do not continue scraping.")
        raise RuntimeError(f"Blocked or rate-limited with status {response.status_code}")

    if response.status_code == 404:
        print(f"[404] Missing interaction ID: {url}")
        return None

    if response.status_code >= 500:
        print(f"[SERVER ERROR {response.status_code}] Skipping: {url}")
        return None

    print(f"[HTTP {response.status_code}] {url}")
    return None


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_drug_names_from_title(title: str) -> tuple[str | None, str | None]:
    pattern = r"Interaction between\s+(.+?)\s+and\s+(.+)$"
    match = re.search(pattern, title, flags=re.IGNORECASE)

    if not match:
        return None, None

    drug_a = clean_text(match.group(1))
    drug_b = clean_text(match.group(2))
    return drug_a, drug_b


def has_real_value(value: str | None) -> bool:
    return bool(value and clean_text(value) != "-")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def print_progress(
    completed: int,
    total: int,
    collected: int,
    started_at: float,
    bar_width: int = 30,
) -> None:
    if total <= 0:
        return

    percent = completed / total
    filled = int(bar_width * percent)
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = time.monotonic() - started_at
    eta = "?"

    if completed:
        seconds_per_item = elapsed / completed
        remaining = (total - completed) * seconds_per_item
        eta = format_duration(remaining)

    print(
        f"[PROGRESS] |{bar}| {completed}/{total} "
        f"({percent * 100:5.1f}%) | rows: {collected} | "
        f"elapsed: {format_duration(elapsed)} | eta: {eta}",
        flush=True,
    )


def parse_interaction_page(html: str, interaction_id: int) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.select_one("div.title")
    title = clean_text(title_el.get_text(" ", strip=True)) if title_el else None

    drug_a, drug_b = extract_drug_names_from_title(title or "")

    severity = None
    mechanism_type = None

    badges = [clean_text(b.get_text(" ", strip=True)) for b in soup.select("span.badge")]
    if badges:
        severity = badges[0] if len(badges) >= 1 else None
        mechanism_type = badges[1] if len(badges) >= 2 else None

    fields = {}

    for row in soup.select("table tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        key = clean_text(cells[0].get_text(" ", strip=True))
        value_cell = cells[1]

        for effect_block in value_cell.select("#effect-block"):
            effect_block.decompose()

        value = clean_text(value_cell.get_text(" ", strip=True))
        fields[key] = value

    return {
        "interaction_id": interaction_id,
        "url": INTERACTION_URL_TEMPLATE.format(interaction_id=interaction_id),
        "title": title,
        "drug_a": drug_a,
        "drug_b": drug_b,
        "severity": severity,
        "mechanism_type": mechanism_type,
        "interaction": fields.get("Interaction"),
        "management": fields.get("Management"),
        "raw_id_field": fields.get("ID"),
        "references": fields.get("References"),
    }


def crawl_id_range(
    start_id: int,
    end_id: int,
    delay_seconds: float = 1.0,
    jitter_seconds: float = 0.5,
    checkpoint_every: int = 25,
    randomize_ids: bool = False,
    random_seed: int | None = None,
    max_results: int | None = None,
) -> pd.DataFrame:
    results = []
    session = requests.Session()
    last_url = BASE_URL + "/"
    rng = random.Random(random_seed)
    interaction_ids = list(range(start_id, end_id + 1))
    total_ids = len(interaction_ids)
    started_at = time.monotonic()

    if randomize_ids:
        rng.shuffle(interaction_ids)

    print_progress(0, total_ids, len(results), started_at)

    for i, interaction_id in enumerate(interaction_ids, start=1):
        url = INTERACTION_URL_TEMPLATE.format(interaction_id=interaction_id)
        print(f"[FETCH] {interaction_id}")

        session.headers.update({"Referer": last_url})

        html = fetch_page(session, url)
        if html:
            last_url = url
            parsed = parse_interaction_page(html, interaction_id)

            if (
                parsed["title"]
                and parsed["severity"] != "Unknown"
                and parsed["mechanism_type"] != "Unknown"
                and has_real_value(parsed["interaction"])
                and has_real_value(parsed["management"])
            ):
                results.append(parsed)
                print(
                    f"  -> OK: {parsed['drug_a']} / {parsed['drug_b']} "
                    f"({parsed['severity']}, {parsed['mechanism_type']})"
                )
                if max_results and len(results) >= max_results:
                    print_progress(i, total_ids, len(results), started_at)
                    print(f"[DONE] Reached max_results={max_results}.")
                    save_results(results)
                    return pd.DataFrame(results)
            else:
                print("  -> Parsed page, but skipped due to unknown or empty fields.")

        if checkpoint_every and i % checkpoint_every == 0:
            save_results(results)

        print_progress(i, total_ids, len(results), started_at)

        sleep_time = delay_seconds + rng.uniform(0, jitter_seconds)
        time.sleep(sleep_time)

    save_results(results)
    return pd.DataFrame(results)


def save_results(results: list[dict]) -> None:
    """
    Saves current results to CSV and JSONL.
    """
    if not results:
        return

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[SAVED] {len(results)} rows -> {OUTPUT_CSV}, {OUTPUT_JSONL}")


def main() -> None:
    global OUTPUT_CSV, OUTPUT_JSONL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-id", type=int, required=True)
    parser.add_argument("--end-id", type=int, required=True)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=0.5)
    parser.add_argument("--max-results", type=int, default=300)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--output-jsonl", default=OUTPUT_JSONL)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    args = parser.parse_args()
    if args.start_id < 1 or args.end_id < args.start_id:
        parser.error("IDs must form a positive, ascending range")
    if args.delay < 0 or args.jitter < 0 or args.max_results < 1:
        parser.error("delays must be non-negative and --max-results positive")
    OUTPUT_CSV, OUTPUT_JSONL = args.output_csv, args.output_jsonl
    if Path(OUTPUT_CSV).resolve() == Path(OUTPUT_JSONL).resolve():
        parser.error("CSV and JSONL must use different paths")
    df = crawl_id_range(
        args.start_id,
        args.end_id,
        delay_seconds=args.delay,
        jitter_seconds=args.jitter,
        randomize_ids=args.randomize,
        random_seed=args.seed,
        max_results=args.max_results,
    )
    print(df.head())


if __name__ == "__main__":
    main()
