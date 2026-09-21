"""Embedded Cortex: no server, no API key.

    pip install "cortexlayer[local]"
    python -m spacy download en_core_web_sm      # optional but recommended
    python examples/local_quickstart.py
"""

import tempfile

from cortexlayer import Memory

with tempfile.TemporaryDirectory() as store:      # use Memory() for a persistent ~/.cortexlayer
    m = Memory(store)

    m.add("Christopher Nolan directed Inception.", user_id="alice")
    m.add("Christopher Nolan was born in London in 1970.", user_id="alice")
    m.add("Alice likes hiking.", user_id="alice")
    print("linking:", m.relink(user_id="alice"))

    for hit in m.search("Who directed Inception?", user_id="alice", limit=1):
        how = "vector match" if hit.via == "direct" else f"linked from {hit.linked_from[:8]}…"
        print(f"- {hit.title}  [{how}]")

    print("bob sees:", m.search("Inception", user_id="bob"))     # isolated: []
