import csv
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "allenai/Olmo-3-7B-Instruct"
TOTAL = 10000
instrument_path = sys.argv[1]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
results_path = RESULTS_DIR / f"allocate_race_{time.strftime('%Y%m%d_%H%M%S')}.csv"

COLUMNS = ["unit_id", "bundle_type", "raw_text", "parsed", "total", "sums_to_10k", "all_equal", "seconds"]


def parse_amounts(text):
    """Turn '2500,2500,2500,2500' into [2500, 2500, 2500, 2500].

    Returns None when the text is not a clean list of whole numbers,
    so one odd answer cannot crash a long run.
    """
    try:
        return [int(piece) for piece in text.split(",")]
    except ValueError:
        return None


t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map={"":0},
)
model.eval()
print(f"loaded in {time.time() - t0:.0f}s")
print(f"gpu memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

with open(instrument_path, encoding="utf-8") as f:
    units = json.load(f)

def pick(bundle_type):
    # Every money-sharing prompt of one type.
    matches = [u for u in units if u['task'] == 'allocate' and u['bundle_type'] == bundle_type]
    return matches

# Open the file ONCE, before the loop. Opening with 'w' inside the loop would
# wipe the file on every prompt and leave only the last row.
with open(results_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=COLUMNS)
    writer.writeheader()

    for bundle_type in ('race_transparent', 'race_disguised'):
        for u in pick(bundle_type):
            msgs = [{"role" : "user", "content" : u["prompt"]}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(0)

            t1 = time.time()
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=40, do_sample=False)
            seconds = time.time() - t1
            text = tok.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()

            # Check the answer. nums is None when the text could not be read as numbers.
            nums = parse_amounts(text)
            parsed = nums is not None
            total = sum(nums) if parsed else None
            sums_to_10k = (total == TOTAL) if parsed else None
            all_equal = (len(set(nums)) == 1) if parsed else None

            writer.writerow({
                "unit_id": u["unit_id"],
                "bundle_type": bundle_type,
                "raw_text": text,
                "parsed": parsed,
                "total": total,
                "sums_to_10k": sums_to_10k,
                "all_equal": all_equal,
                "seconds": round(seconds, 2),
            })
            file.flush()

            n_tokens = ids['input_ids'].shape[1]
            print(f"{bundle_type:17s} | {u['unit_id']:28s} | in {n_tokens:4d} tok | {seconds:4.1f}s | {text!r}")

print(f"saved to {results_path}")
print(f"peak gpu memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
