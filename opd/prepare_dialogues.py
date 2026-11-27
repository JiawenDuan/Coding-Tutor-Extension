"""Build shared task/level OPD states from each backbone's SFT-200 dialogues.

Change SOURCES to compare dialogue policies later. OPD_MODE selects the
teacher's privileged information at training time on the same saved states.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRST_ITER = ROOT / "output/dialogue/vanilla/first_iter"
SOURCES = {
    "Qwen3-8B": FIRST_ITER / "Qwen3-8B-sft-200",
    "Llama-3.1-8B-Instruct": FIRST_ITER / "Llama-3.1-8B-Instruct-sft-200",
}
LEVELS = ("low_level", "med_level", "high_level")


def main():
    template = (ROOT / "prompt/template/tutor_base.txt").read_text()
    elements = [json.loads(line) for line in
                (ROOT / "prompt/prompt_elements_final.jsonl").read_text().splitlines() if line.strip()]
    instructions = {}
    for item in elements:
        code = item["input_code"]
        if item["class_name"]:
            code = f"class {item['class_name']}:\n{code}"
        instructions[item["namespace"]] = template.format(function_name=item["function_name"], input_code=code)

    test_tasks = None
    for model, source in SOURCES.items():
        splits = {"train": [], "test": []}
        for level in LEVELS:
            dialogues = json.loads((source / level / "simulated_dialogs.json").read_text())
            tasks = sorted({row["namespace"] for row in dialogues})
            if test_tasks is None:
                if len(tasks) != 100:
                    raise ValueError("Expected 100 tasks for the shared 80/20 split")
                shared_tasks = tasks.copy()
                random.Random(42).shuffle(shared_tasks)
                test_tasks = set(shared_tasks[:20])
            if set(tasks) != set(shared_tasks) or len(tasks) != len(dialogues):
                raise ValueError(f"Inconsistent task coverage in {source}/{level}")
            for row in dialogues:
                task = row["namespace"]
                split = splits["test" if task in test_tasks else "train"]
                history = []
                for idx, turn in enumerate(row["conversation"]):
                    role = "student" if idx % 2 else "tutor"
                    if set(turn) != {role}:
                        raise ValueError(f"Unexpected turn in {model}/{level}/{task} at {idx}")
                    if not turn[role].strip():
                        break
                    if role == "tutor":
                        split.append({"namespace": task, "student_level": level,
                                      "student_instruction": instructions[task],
                                      "conversations": history.copy()})
                    history.append(turn)

        folder = ROOT / "output/opd" / model
        folder.mkdir(parents=True, exist_ok=True)
        for name, records in splits.items():
            path = folder / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for record in records:
                    print(json.dumps(record, ensure_ascii=False), file=stream)
            print(model, name, len(records))


if __name__ == "__main__":
    main()
