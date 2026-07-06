#!/usr/bin/env python3
"""Simple concurrent benchmark for the RunPod serverless endpoint.

Uses only `requests` (no aiohttp) to keep dependencies minimal.

Usage:
    export RUNPOD_API_KEY=your_key_here
    python3 benchmark.py --concurrency 10 --requests 100
    python3 benchmark.py --concurrency 10 --duration 30m

RunPod serverless /run returns a job ID immediately; this script optionally
polls /status/{id} until each job completes.
"""

import argparse
import os
import random
import string
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait

import requests

ENDPOINT_URL = "https://api.runpod.ai/v2/34orgs5ae40zo7"


def parse_duration(value: str | None) -> float | None:
    """Parse a duration string like '30m', '1h', '1800s', or a raw number of seconds.

    Returns duration in seconds, or None if value is None/empty.
    """
    if value is None or value == "":
        return None
    value = value.strip().lower()
    if not value:
        return None

    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    for suffix, multiplier in multipliers.items():
        if value.endswith(suffix):
            number_part = value[:-1].strip()
            if not number_part:
                raise argparse.ArgumentTypeError(f"Invalid duration: {value!r}")
            return float(number_part) * multiplier

    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid duration: {value!r}. Use e.g. 1800, 30m, 1h, 1d."
        ) from exc


def get_auth_header() -> dict[str, str]:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the RUNPOD_API_KEY environment variable.")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def make_payload(prompt: str, max_tokens: int) -> dict:
    return {
        "input": {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 1.0,
        }
    }


def random_payload(
    min_prompt_tokens: int = 10,
    max_prompt_tokens: int = 200,
    min_max_tokens: int = 16,
    max_max_tokens: int = 512,
) -> dict:
    """Generate a random payload so each request differs inside the script."""
    prompt_length = random.randint(min_prompt_tokens, max_prompt_tokens)
    # ~5 characters per token is a rough heuristic for English-like text.
    text = "".join(
        random.choices(
            string.ascii_letters + string.digits + string.punctuation + " ",
            k=prompt_length * 5,
        )
    )
    return make_payload(
        prompt=f"Summarize the following in one sentence: {text}",
        max_tokens=random.randint(min_max_tokens, max_max_tokens),
    )


def submit_one(session: requests.Session, payload: dict, timeout: int) -> dict:
    start = time.monotonic()
    try:
        resp = session.post(f"{ENDPOINT_URL}/run", json=payload, timeout=timeout)
        return {
            "status": resp.status_code,
            "latency": time.monotonic() - start,
            "data": resp.json() if resp.text else None,
        }
    except Exception as exc:
        return {
            "status": None,
            "latency": time.monotonic() - start,
            "error": str(exc),
        }


def poll_status(session: requests.Session, job_id: str, timeout: int, poll_interval: float = 1.0) -> dict:
    url = f"{ENDPOINT_URL}/status/{job_id}"
    start = time.monotonic()
    while True:
        if time.monotonic() - start > timeout:
            return {"completed": False, "error": "polling timeout"}
        try:
            resp = session.get(url, timeout=30)
            data = resp.json()
            status = data.get("status")
            if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                return {
                    "completed": True,
                    "status": status,
                    "total_time": time.monotonic() - start,
                    "data": data,
                }
        except Exception as exc:
            return {"completed": False, "error": str(exc)}
        time.sleep(poll_interval)


def run_one(
    headers: dict[str, str],
    payload: dict,
    poll: bool,
    submit_timeout: int,
    poll_timeout: int,
) -> dict:
    session = requests.Session()
    session.headers.update(headers)

    result = submit_one(session, payload, submit_timeout)

    if poll and result.get("status") == 200:
        job_id = result.get("data", {}).get("id")
        if job_id:
            result["poll"] = poll_status(session, job_id, poll_timeout)

    return result


def main(
    concurrency: int,
    prompt: str,
    max_tokens: int,
    poll: bool,
    submit_timeout: int,
    poll_timeout: int,
    random_payloads: bool = False,
    total_requests: int | None = None,
    duration: float | None = None,
) -> None:
    if total_requests is None and duration is None:
        raise ValueError("Specify either --requests or --duration.")
    if total_requests is not None and duration is not None:
        raise ValueError("Specify either --requests or --duration, not both.")

    headers = get_auth_header()

    def next_payload() -> dict:
        if random_payloads:
            return random_payload()
        return make_payload(prompt, max_tokens)

    results: list[dict] = []
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        if total_requests is not None:
            # Fixed request count mode
            futures = {
                executor.submit(run_one, headers, next_payload(), poll, submit_timeout, poll_timeout)
                for _ in range(total_requests)
            }
            for future in as_completed(futures):
                results.append(future.result())
        else:
            # Duration mode: keep up to `concurrency` requests in flight until time is up
            futures: set = set()
            submitted = 0
            running = True

            while running or futures:
                # Refill the worker pool while there is still time
                while running and len(futures) < concurrency:
                    if time.monotonic() - start >= duration:
                        running = False
                        break
                    futures.add(
                        executor.submit(run_one, headers, next_payload(), poll, submit_timeout, poll_timeout)
                    )
                    submitted += 1

                if not futures:
                    break

                done, futures = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    results.append(future.result())

    total_time = time.monotonic() - start
    actual_requests = len(results)

    # Summary
    successes = sum(1 for r in results if r.get("status") == 200)
    failures = actual_requests - successes
    submit_latencies = [r["latency"] for r in results if "latency" in r]

    print(f"Total requests: {actual_requests}")
    print(f"Concurrency: {concurrency}")
    if duration is not None:
        print(f"Target duration: {duration:.0f}s ({duration / 60:.2f}m)")
    print(f"Successful /run submits: {successes}")
    print(f"Failed /run submits: {failures}")
    print(f"Total wall time: {total_time:.2f}s")
    print(f"Throughput: {actual_requests / total_time:.2f} req/s")
    if submit_latencies:
        avg = sum(submit_latencies) / len(submit_latencies)
        print(f"Submit latency avg: {avg:.3f}s")
        print(f"Submit latency min: {min(submit_latencies):.3f}s")
        print(f"Submit latency max: {max(submit_latencies):.3f}s")

    if poll:
        completed_polls = [
            r["poll"]
            for r in results
            if isinstance(r.get("poll"), dict) and r["poll"].get("completed")
        ]
        print(f"Completed poll results: {len(completed_polls)}")
        if completed_polls:
            poll_times = [p["total_time"] for p in completed_polls]
            avg_poll = sum(poll_times) / len(poll_times)
            print(f"End-to-end time avg: {avg_poll:.3f}s")
            print(f"End-to-end time max: {max(poll_times):.3f}s")

    if failures:
        print("\nFirst failure:")
        for r in results:
            if r.get("status") != 200:
                print(r)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RunPod serverless benchmark")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Total number of requests to send (mutually exclusive with --duration)",
    )
    parser.add_argument(
        "--duration",
        type=parse_duration,
        default=None,
        metavar="DURATION",
        help="Run for a duration such as 30m, 1h, 1800s, or 0.5h (mutually exclusive with --requests)",
    )
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--poll", action="store_true", help="Poll /status until each job completes")
    parser.add_argument("--submit-timeout", type=int, default=60)
    parser.add_argument("--poll-timeout", type=int, default=300)
    parser.add_argument(
        "--random",
        action="store_true",
        dest="random_payloads",
        help="Generate a random prompt and max_tokens for every request inside the script",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible random payloads",
    )
    args = parser.parse_args()

    if args.requests is None and args.duration is None:
        args.requests = 100

    if args.random_payloads and args.random_seed is not None:
        random.seed(args.random_seed)

    main(
        concurrency=args.concurrency,
        total_requests=args.requests,
        duration=args.duration,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        poll=args.poll,
        submit_timeout=args.submit_timeout,
        poll_timeout=args.poll_timeout,
        random_payloads=args.random_payloads,
    )
