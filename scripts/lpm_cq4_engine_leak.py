import json
from cactus.bindings import cactus as c

BUNDLE = "/work/bundle"
RES_LO, RES_HI = 151670, 151925

model = c.cactus_init(BUNDLE)
print("init ok")

# reserved-ID transport check (M1 #186 method: special_tokens.json)
print("tokenize hello <|mem0_0|>:", c.cactus_tokenize(model, "hello <|mem0_0|>"))
print("tokenize mem0_0 mem1_0:", c.cactus_tokenize(model, "<|mem0_0|> <|mem1_0|>"))


def generate(prompt, max_tokens=16):
    ids = []

    def cb(text, token_id):
        ids.append(int(token_id))

    res = c.cactus_complete(
        model,
        [{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "max_tokens": max_tokens},
        callback=cb,
    )
    if not isinstance(res, dict):
        res = {"raw": str(res)}
    return res, ids


for base in ["The best fertilizer for tomatoes is", "One fact about soil:"]:
    variants = [
        ("none", base),
        ("1 mem", base + " <|mem0_0|>"),
        ("8 mem", base + " " + " ".join(f"<|mem{i}_0|>" for i in range(8))),
    ]
    for label, prompt in variants:
        res, ids = generate(prompt)
        text = res.get("response") or res.get("completion") or res.get("text") or json.dumps(res)[:200]
        reserved = [i for i in ids if RES_LO <= i <= RES_HI]
        print(
            f"\n[{label}] prompt={prompt[:60]!r}\n"
            f"  keys={list(res.keys())[:6]}\n"
            f"  text={str(text)[:140]!r}\n"
            f"  gen_ids={ids}\n"
            f"  reserved_emitted={len(reserved)}/{len(ids)} {reserved[:8]}"
        )

c.cactus_destroy(model)
