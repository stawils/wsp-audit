#!/usr/bin/env python3
"""Self-test for audit/audit.py — asserts the numbers we independently established, on known-good
and known-bad input, before the kit's verdicts are allowed to count for anything else."""
import json, subprocess, sys

PY = ".venv/bin/python"
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


print("verify — exp 8 released bytes")
subprocess.run([PY, "audit/audit.py", "verify", "--json-out", "/tmp/v.json"], check=True,
               capture_output=True)
v = json.load(open("/tmp/v.json"))["verify"]
check("rebuilds to the published sha256", v["checks"]["rebuilt"])
check("addressed 131,072 ≠ changed 130,188",
      (v["totals"]["addressed"], v["totals"]["changed"]) == (131072, 130188), str(v["totals"]))
check("884 entries round back to bf16", v["totals"]["addressed"] - v["totals"]["changed"] == 884)
check("one constant dose, layer 15", abs(v["tensors"][0]["alpha_mean"] - 6.0006) < 0.01
      and v["tensors"][0]["alpha_rel_spread"] < 0.01, str(v["tensors"][0]["alpha_mean"]))
check("one constant dose, layer 19", abs(v["tensors"][1]["alpha_mean"] - 6.0004) < 0.01)
check("64 columns per layer", all(t["columns"] == 64 for t in v["tensors"]))

print("\nverify — the other two released edits (different claim shapes)")
for run, want in (("p10c", dict(tensors=30, form="lowrank", rank=1)),
                  ("p25c", dict(tensors=1, form="sparse", zero_cols=256))):
    d = f"sources/lm_understand-logs/exp8_math/runs/{run}/edit"
    subprocess.run([PY, "audit/audit.py", "verify", "--edit", d, "--json-out", f"/tmp/{run}.json"],
                   check=True, capture_output=True)
    r = json.load(open(f"/tmp/{run}.json"))["verify"]
    check(f"{run}: rebuilds to the published sha256", r["checks"]["rebuilt"])
    check(f"{run}: counts match the manifest", r["checks"]["counts_match_manifest"])
    check(f"{run}: {want['tensors']} tensors", r["totals"]["tensors"] == want["tensors"],
          str(r["totals"]["tensors"]))
    check(f"{run}: form is {want['form']}", all(t["form"] == want["form"] for t in r["tensors"]))
    if want["form"] == "lowrank":
        check(f"{run}: every tensor rank {want['rank']}",
              all(t.get("rank") == want["rank"] for t in r["tensors"]))
    else:
        check(f"{run}: {want['zero_cols']} columns set to zero",
              r["tensors"][0]["columns"] == want["zero_cols"]
              and r["tensors"][0]["entries_set_to_zero"] == r["tensors"][0]["changed_entries"])

print("\ncases — the four outcomes we measured ourselves")
subprocess.run([PY, "audit/audit.py", "cases", "--json-out", "/tmp/c.json"], check=True,
               capture_output=True)
rows = {r["cell"]: r for r in json.load(open("/tmp/c.json"))["cases"]}
for cell, want in (("gpt2_L11", True), ("gpt2_L6", False),
                   ("distilgpt2_L5", False), ("Qwen_Qwen2.5-0.5B_L23", True)):
    check(f"{cell} reproduces {'works' if want else 'fails'}", rows[cell]["success"] == want,
          f"hit {rows[cell]['hit']:.0%} false {rows[cell]['false']:.0%}")
check("11 cells on record", len(rows) == 11, str(len(rows)))
check("7 of 11 work", sum(r["success"] for r in rows.values()) == 7)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(0 if not fails else 1)
