'''
HTTP server for the food-detection pipeline.

The app has two ways to get a reading, and the server serves both. The
underlying VL/RAM++ -> food filter -> GroundingDINO pipeline tags foods (via a
vision LLM when available, else RAM++), detects them, and scores freshness --
lychee and shrimp with real vision models, other foods with VL spoilage on each
crop:

  1. AUTOMATIC (every 2 hours): the RAM++ -> food filter -> GroundingDINO
     pipeline runs on a background timer against the newest camera snapshot
     (collect/data/latest.jpg). The result is cached in memory and
     written to outputs/latest/detection_results.json, which holds only the
     newest run. The app reads this cached result from GET /latest instantly --
     no inference on the request path.

  2. MANUAL (the "detect" button): GET/POST http://<host>:<port>/detect runs
     the whole pipeline right then and returns the fresh summary in that
     response. It DOES update the /latest cache (so /latest reflects it,
     trigger:"manual"), but it does NOT push to /stream -- the button's result
     reaches the app via its own HTTP response, not as an SSE event.

Both channels refresh the /latest cache and append to outputs/history.jsonl (one
JSON object per line) so past readings survive. /latest reflects the most recent
run of either kind; use the trigger field ("scheduled"|"manual") to tell them
apart. Only AUTOMATIC runs are PUSHED to GET /stream (Server-Sent Events), so
/stream is purely the automatic feed.

Both models are loaded ONCE at startup and kept in memory, so each run only does
inference instead of reloading ~6 GB of weights.

Run it (leave it running; the app talks to it over HTTP):
  /opt/miniconda3/envs/food_pipe/bin/python detect_server.py --cpu-only \
      --host 0.0.0.0 --port 4100

Endpoints:
  GET  /health         -> status, device, cache age, seconds to next run
  GET|POST /detect     -> run the pipeline now, return fresh summary; updates
                          /latest but does not push to /stream (slow)
  GET  /latest         -> the cached summary from the last run (instant)
  GET  /stream         -> SSE: pushes each automatic run's summary as it finishes
  GET  /history        -> every recorded run, oldest first
  GET  /history?limit=10&since=2026-07-17 -> the tail / runs after a timestamp
'''
import argparse
import collections
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch

# food_pipeline puts vendor RAM++ / GroundingDINO on sys.path on import
import food_pipeline as fp
from ram.models import ram_plus
from groundingdino.util.inference import load_model

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- configuration (overridable via CLI) -----------------------------------
CONFIG = {
    'image': os.path.join(_HERE, 'collect', 'data', 'latest.jpg'),
    'output_dir': os.path.join(_HERE, 'outputs', 'latest'),
    'ram_checkpoint': os.path.join(
        _HERE, 'vendor', 'recognize-anything', 'pretrained',
        'ram_plus_swin_large_14m.pth'),
    'gdino_checkpoint': os.path.join(
        _HERE, 'vendor', 'GroundingDINO', 'weights',
        'groundingdino_swint_ogc.pth'),
    'gdino_config': fp._DEFAULT_GDINO_CFG,
    'image_size': 384,
    'box_threshold': 0.3,
    'text_threshold': 0.25,
    # 每种食物最多保留几个框；0/None = 不限制
    'max_per_food': 0,
    'device': 'cpu',
    'interval': 2 * 60 * 60,   # seconds between scheduled runs (2 hours)
    # Append-only log of every run. outputs/latest/ only ever holds the newest
    # result; this keeps the whole history, one JSON object per line.
    'history_file': os.path.join(_HERE, 'outputs', 'history.jsonl'),
    # Food tagger: auto = VL API then RAM++; vl = API only; ram = RAM++ only.
    'tagger': 'auto',
}

# Models are shared across requests; the lock serializes inference because a
# single model is not safe to run concurrently from multiple threads.
_MODELS = {'ram': None, 'gdino': None, 'lychee': None, 'shrimp': None}
_LOCK = threading.Lock()

# The most recent good result, served to every /detect caller. _CACHE_LOCK
# guards it because the scheduler thread writes while request threads read.
_CACHE = {
    'summary': None,        # last successful summary dict
    'generated_ts': None,   # epoch seconds when that summary was produced
    'next_run_ts': None,    # epoch seconds of the next scheduled run
    'last_error': None,     # error from the most recent run, if any
    'trigger': None,        # what produced it: 'scheduled' or 'manual'
}
_CACHE_LOCK = threading.Lock()
_STOP = threading.Event()   # set on shutdown to end the scheduler thread

# A history line can exceed the size the OS appends atomically, so serialize
# writers rather than relying on O_APPEND to keep lines from interleaving.
_HISTORY_LOCK = threading.Lock()

# Open /stream (SSE) connections. Each is a bounded queue the client's handler
# thread drains; a finished run drops the new summary into every queue so the
# app is pushed the result instead of having to poll /latest. _SUBS_LOCK guards
# the set because the scheduler broadcasts while request threads add/remove.
_SUBSCRIBERS = set()
_SUBS_LOCK = threading.Lock()


def load_models():
    print(f"Loading RAM++ on {CONFIG['device']} (this takes a moment) ...")
    _MODELS['ram'] = ram_plus(
        pretrained=CONFIG['ram_checkpoint'],
        image_size=CONFIG['image_size'],
        vit='swin_l').eval().to(CONFIG['device'])
    print('Loading GroundingDINO ...')
    _MODELS['gdino'] = load_model(
        CONFIG['gdino_config'], CONFIG['gdino_checkpoint'],
        device=CONFIG['device'])
    print('Loading LycheePredictor (DINOv2 + multitask head) ...')
    try:
        _MODELS['lychee'] = fp.get_lychee_predictor(device=CONFIG['device'])
        ver = getattr(_MODELS['lychee'], 'model_version', '?')
        print(f'  lychee freshness ready ({ver}).')
    except Exception as exc:
        _MODELS['lychee'] = None
        print(f'  lychee predictor unavailable: {exc}')
    print('Loading ShrimpPredictor (EfficientNet-B0 multitask) ...')
    try:
        _MODELS['shrimp'] = fp.get_shrimp_predictor(device=CONFIG['device'])
        ver = getattr(_MODELS['shrimp'], 'model_version', '?')
        print(f'  shrimp freshness ready ({ver}).')
    except Exception as exc:
        _MODELS['shrimp'] = None
        print(f'  shrimp predictor unavailable: {exc}')
    print('Models ready.')


def run_detection():
    """Run the pipeline on the current latest.jpg and return the summary dict."""
    output_dir = CONFIG['output_dir']
    image = CONFIG['image']
    if not os.path.exists(image):
        return {'error': f'image not found: {image}'}

    device = CONFIG['device']
    _all_tags, foods, meta = fp.recognize_food_tags(
        image, ram_model=_MODELS['ram'], image_size=CONFIG['image_size'],
        device=device, tagger=CONFIG.get('tagger', 'auto'))

    if not foods:
        detections = []
    else:
        # VL API counts take priority; CONFIG max_per_food is uniform fallback.
        max_per_food = meta.get('food_counts') or CONFIG.get('max_per_food') or None
        _counts, detections = fp.detect_and_crop(
            image, foods, _MODELS['gdino'], output_dir, device,
            CONFIG['box_threshold'], CONFIG['text_threshold'],
            max_per_food=max_per_food)

    results = fp.build_detection_results(
        image, detections, label_zh=meta.get('label_zh'),
        output_dir=output_dir,
        lychee_predictor=_MODELS.get('lychee'),
        shrimp_predictor=_MODELS.get('shrimp'),
        device=device)
    results['tagger'] = meta.get('source')
    if meta.get('food_counts'):
        results['expectedCounts'] = meta['food_counts']

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'detection_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def _log(msg):
    """Print a timestamped line, flushed so it survives log redirection.

    The server runs unattended for hours, usually with stdout pointed at a
    file, where Python's block buffering would otherwise hide these for ages.
    """
    print(f'{_iso(time.time())} {msg}', flush=True)


def _iso(ts):
    """Epoch seconds -> local ISO-8601 string (None passes through)."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).astimezone().isoformat(
        timespec='seconds')


def append_history(summary, trigger, ts):
    """Append one run to the history file as a single JSON line.

    outputs/latest/detection_results.json is overwritten by every run, so this
    is the only place a past reading survives. Failed runs are not recorded --
    the gap between timestamps is what shows a run was missed.
    """
    path = CONFIG['history_file']
    if not path:
        return
    record = {
        'timestamp': _iso(ts),
        'trigger': trigger,                  # 'scheduled' or 'manual'
        'isSpoiled': summary.get('isSpoiled'),
        'message': summary.get('message'),
        'detectedFoods': summary.get('detectedFoods', []),
    }
    line = json.dumps(record, ensure_ascii=False) + '\n'
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _HISTORY_LOCK:                  # keep concurrent lines intact
            with open(path, 'a', encoding='utf-8') as f:
                f.write(line)
    except OSError as exc:                   # history is never worth a crash
        _log(f'[history] could not append to {path}: {exc}')


def read_history(limit=None, since=None):
    """Return history records, oldest first, optionally the last `limit` only."""
    path = CONFIG['history_file']
    if not path or not os.path.exists(path):
        return []
    records = []
    with open(path, encoding='utf-8') as f:
        # deque keeps only the tail in memory, so this stays cheap as the file
        # grows past thousands of runs.
        lines = collections.deque(f, maxlen=limit) if limit else f
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:               # skip a torn/partial line
                continue
    if since:
        records = [r for r in records if (r.get('timestamp') or '') >= since]
    return records


def _sse_frame(payload, event='detection'):
    """Serialize a payload as one Server-Sent Events frame."""
    return (f'event: {event}\n'
            f'data: {json.dumps(payload, ensure_ascii=False)}\n\n')


def _broadcast(payload):
    """Push a payload to every open /stream connection.

    Runs from the thread that finished a run. A slow client whose queue is full
    is skipped rather than blocking the broadcast -- it will still catch up on
    its next /latest poll, and gets the current snapshot when it reconnects.
    """
    frame = _sse_frame(payload)
    with _SUBS_LOCK:
        subscribers = list(_SUBSCRIBERS)
    delivered = 0
    for q in subscribers:
        try:
            q.put_nowait(frame)
            delivered += 1
        except queue.Full:                # slow client; it'll catch up on /latest
            pass
    _log(f'[stream] pushed detection to {delivered}/{len(subscribers)} '
         f'subscriber(s)')


def refresh_cache(trigger='scheduled', broadcast=True):
    """Run the pipeline once and cache the result. Returns the summary dict.

    Both automatic and manual runs refresh the /latest cache (so /latest always
    reflects the newest reading). Only automatic runs push to /stream: a manual
    /detect passes broadcast=False so the button's result reaches the app via
    its own HTTP response, not as an SSE event.

    A failed run leaves the previous good summary in place -- a stale answer is
    more useful to the app than an error, so failures are only recorded in
    'last_error' and surfaced through /health.
    """
    started = time.time()
    with _LOCK:                       # one inference at a time
        try:
            summary = run_detection()
        except Exception as exc:      # keep the scheduler alive across failures
            summary = {'error': str(exc)}

    error = summary.get('error')
    finished = time.time()
    with _CACHE_LOCK:
        _CACHE['last_error'] = error
        if not error:
            _CACHE['summary'] = summary
            _CACHE['generated_ts'] = finished
            _CACHE['trigger'] = trigger
    if not error:
        append_history(summary, trigger, finished)
        if broadcast:
            # Push the fresh result to every open /stream connection.
            # cached_summary reads the cache we just wrote, so it carries the
            # same generated_at / age_seconds fields the app sees from /latest.
            body, _status = cached_summary()
            _broadcast(body)

    took = time.time() - started
    label = 'scheduler' if trigger == 'scheduled' else 'manual'
    if error:
        _log(f'[{label}] detection failed after {took:.1f}s: {error}')
    else:
        n = len(summary.get('detectedFoods', []))
        _log(f'[{label}] detection done in {took:.1f}s -- {n} item(s)')
    return summary


def seed_cache_from_disk():
    """Serve the previous run's result until the first scheduled run lands.

    Without this the app would get a 503 for the minutes it takes the first run
    to finish after a restart, even though a perfectly good result is on disk.
    """
    path = os.path.join(CONFIG['output_dir'], 'detection_results.json')
    try:
        with open(path, encoding='utf-8') as f:
            summary = json.load(f)
    except (OSError, ValueError) as exc:
        _log(f'[cache] no previous result to seed ({exc})')
        return
    with _CACHE_LOCK:
        _CACHE['summary'] = summary
        _CACHE['generated_ts'] = os.path.getmtime(path)
        # The persisted file doesn't record which run wrote it; treat a
        # seeded result as the automatic baseline until a real run replaces it.
        _CACHE['trigger'] = 'scheduled'
    _log(f'[cache] seeded from {path} (written {_iso(_CACHE["generated_ts"])})')


def cached_summary():
    """The cached summary plus freshness metadata, and an HTTP status."""
    now = time.time()
    with _CACHE_LOCK:
        summary = _CACHE['summary']
        generated_ts = _CACHE['generated_ts']
        next_run_ts = _CACHE['next_run_ts']
        last_error = _CACHE['last_error']
        trigger = _CACHE['trigger']

    if summary is None:
        return {'error': 'no detection result available yet; the first run is '
                         'still in progress',
                'last_error': last_error}, 503

    body = dict(summary)
    body['generated_at'] = _iso(generated_ts)
    body['age_seconds'] = int(now - generated_ts) if generated_ts else None
    # 'manual' = produced by a /detect button press; 'scheduled' = the 2h timer.
    # Lets the app label a reading "manual check" vs "automatic check".
    body['trigger'] = trigger
    if next_run_ts:
        body['next_run_in_seconds'] = max(int(next_run_ts - now), 0)
    if last_error:
        body['last_error'] = last_error
    return body, 200


def _scheduler():
    """Re-run the pipeline every CONFIG['interval'] seconds until shutdown."""
    while not _STOP.is_set():
        refresh_cache()
        with _CACHE_LOCK:
            _CACHE['next_run_ts'] = time.time() + CONFIG['interval']
        _log(f"[scheduler] next run at {_iso(_CACHE['next_run_ts'])}")
        _STOP.wait(CONFIG['interval'])   # interruptible sleep


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        """Server-Sent Events: hold the connection open and push automatic runs.

        The app opens this once (EventSource) and receives a 'detection' event
        when each scheduled run finishes (manual /detect does not push here),
        carrying the same JSON /latest returns. The current cached result is
        sent immediately on connect so a just-opened app is never blank. A
        ':keepalive' comment goes out every 15s so idle connections (and dead
        ones) are noticed.
        """
        q = queue.Queue(maxsize=8)
        with _SUBS_LOCK:
            _SUBSCRIBERS.add(q)
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            # Send the current snapshot right away (may be a 503-shaped body
            # before the first run lands; the app can read that too).
            body, _status = cached_summary()
            self.wfile.write(_sse_frame(body).encode('utf-8'))
            self.wfile.flush()

            while not _STOP.is_set():
                try:
                    frame = q.get(timeout=15)
                except queue.Empty:
                    frame = ': keepalive\n\n'   # heartbeat / dead-peer probe
                self.wfile.write(frame.encode('utf-8'))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass                                # client hung up; just clean up
        finally:
            with _SUBS_LOCK:
                _SUBSCRIBERS.discard(q)

    def _handle(self):
        parsed = urlparse(self.path)
        route = parsed.path.rstrip('/')
        query = parse_qs(parsed.query)

        if route == '/health':
            now = time.time()
            with _CACHE_LOCK:
                generated_ts = _CACHE['generated_ts']
                next_run_ts = _CACHE['next_run_ts']
                last_error = _CACHE['last_error']
            with _SUBS_LOCK:
                subscribers = len(_SUBSCRIBERS)
            self._send_json({
                'status': 'ok',
                'device': CONFIG['device'],
                'interval_seconds': CONFIG['interval'],
                'has_result': generated_ts is not None,
                'generated_at': _iso(generated_ts),
                'age_seconds': int(now - generated_ts) if generated_ts else None,
                'next_run_in_seconds': (max(int(next_run_ts - now), 0)
                                        if next_run_ts else None),
                'stream_subscribers': subscribers,
                'last_error': last_error,
                'tagger': CONFIG.get('tagger', 'auto'),
            })
        elif route == '/detect':
            # The manual "detect" button: run the whole pipeline now. It updates
            # the /latest cache (trigger "manual") so /latest reflects it, but
            # passes broadcast=False -- the result reaches the app via THIS
            # response, not as an SSE push. /stream stays the automatic feed.
            summary = refresh_cache(trigger='manual', broadcast=False)
            if summary.get('error'):
                self._send_json({'error': summary['error']}, status=500)
                return
            body, status = cached_summary()
            self._send_json(body, status=status)
        elif route == '/latest':
            # The automatic 2-hour channel: hand back the last cached result
            # immediately, no inference on the request path.
            body, status = cached_summary()
            self._send_json(body, status=status)
        elif route == '/stream':
            self._stream()
        elif route == '/history':
            try:
                limit = int(query['limit'][0]) if query.get('limit') else None
            except ValueError:
                self._send_json({'error': 'limit must be an integer'},
                                status=400)
                return
            since = query['since'][0] if query.get('since') else None
            records = read_history(limit=limit, since=since)
            self._send_json({'count': len(records), 'runs': records})
        else:
            self._send_json({'error': 'not found',
                             'routes': ['/detect', '/latest', '/stream',
                                        '/health', '/history']},
                            status=404)

    do_GET = _handle
    do_POST = _handle

    def log_message(self, fmt, *args):        # quieter console
        print('[server]', self.address_string(), fmt % args)


def main():
    parser = argparse.ArgumentParser(description='Food-detection HTTP server')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--image', help='override path to the snapshot image')
    parser.add_argument('--output-dir', help='override output directory')
    parser.add_argument('--gdino-checkpoint', help='override GDINO checkpoint')
    parser.add_argument('--ram-checkpoint', help='override RAM++ checkpoint')
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--box-threshold', type=float)
    parser.add_argument('--text-threshold', type=float)
    parser.add_argument(
        '--max-per-food', type=int, default=None,
        help='每种食物最多保留几个检测框（按置信度；0=不限制）')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='hours between scheduled runs (default: 2)')
    parser.add_argument('--no-scheduler', action='store_true',
                        help='do not run on a timer; /latest updates only when '
                             '/detect is pressed')
    parser.add_argument('--history-file',
                        help='append-only log of every run '
                             '(default: outputs/history.jsonl)')
    parser.add_argument('--no-history', action='store_true',
                        help='do not record run history')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument(
        '--tagger', choices=['auto', 'vl', 'ram'], default='auto',
        help='food tagger: auto=VL API then RAM++, vl=API only, ram=RAM++ only')
    args = parser.parse_args()

    if args.image:
        CONFIG['image'] = args.image
    if args.output_dir:
        CONFIG['output_dir'] = args.output_dir
    if args.gdino_checkpoint:
        CONFIG['gdino_checkpoint'] = args.gdino_checkpoint
    if args.ram_checkpoint:
        CONFIG['ram_checkpoint'] = args.ram_checkpoint
    if args.image_size:
        CONFIG['image_size'] = args.image_size
    if args.box_threshold is not None:
        CONFIG['box_threshold'] = args.box_threshold
    if args.text_threshold is not None:
        CONFIG['text_threshold'] = args.text_threshold
    if args.max_per_food is not None:
        CONFIG['max_per_food'] = args.max_per_food
    if args.history_file:
        CONFIG['history_file'] = args.history_file
    if args.no_history:
        CONFIG['history_file'] = None
    CONFIG['interval'] = args.interval * 3600
    CONFIG['tagger'] = args.tagger
    CONFIG['device'] = ('cpu' if args.cpu_only or not torch.cuda.is_available()
                        else 'cuda')

    load_models()
    seed_cache_from_disk()

    if not args.no_scheduler:
        threading.Thread(target=_scheduler, daemon=True,
                         name='detect-scheduler').start()
        print(f'Scheduler started: running every {args.interval}h '
              f'(first run now).')

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Serving on http://{args.host}:{args.port}  '
          f'(GET/POST /detect = fresh run, GET /latest = cached, '
          f'GET /stream = push, GET /health)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down.')
        _STOP.set()
        server.shutdown()


if __name__ == '__main__':
    main()
