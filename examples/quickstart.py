"""Minimal end-to-end use of cortexlayer.

    export CORTEX_API_KEY=...            # from the Cortex web app → Keys
    python examples/quickstart.py
"""

from cortexlayer import CortexClient, CortexError, WritesNotSupportedError

with CortexClient() as client:
    print("signed in as", client.me().account_name)

    try:
        client.add("Alice moved to Lisbon in March.")
        client.relink()
    except WritesNotSupportedError:
        print("(this server is too old for REST writes — skipping add)")

    try:
        for hit in client.search("Where does Alice live?", limit=3):
            tag = "direct" if hit.via == "direct" else f"via link from {hit.linked_from}"
            print(f"- {hit.title}  [{tag}]")
    except CortexError as e:
        print("search failed:", e)
