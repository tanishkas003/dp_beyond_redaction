"""Local-only adapters; no dataset is downloaded during tests."""
import json
import csv
from dataclasses import dataclass
from pathlib import Path
@dataclass(frozen=True)
class EvaluationExample: text: str; entities: tuple = (); source: str = "synthetic"
def synthetic_examples(): return [EvaluationExample("Contact Priya at priya@example.com.")]
def load_json_examples(path,source="json"):
    raw=json.loads(Path(path).read_text(encoding="utf-8")); rows=raw if isinstance(raw,list) else raw.get("data",[])
    for row in rows:
        text=row.get("text") or row.get("sentence") or row.get("content")
        if text: yield EvaluationExample(text,tuple(row.get("entities",[]) or ()),source)
def load_enron_lines(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip(): yield EvaluationExample(line,source="enron")

def load_tab(path):
    """Load a local Text Anonymization Benchmark-shaped JSON file."""
    return load_json_examples(path, source="TAB")

def load_ai4privacy(path):
    """Load a local AI4Privacy-shaped JSON file."""
    return load_json_examples(path, source="AI4Privacy")

def load_enron(path):
    """Load local Enron text or CSV exports without downloading data."""
    if str(path).lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                text = row.get("body") or row.get("text") or row.get("content")
                if text:
                    yield EvaluationExample(text, source="Enron")
    else:
        yield from load_enron_lines(path)
