import json, os, sys
d, dst = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(d, "_index.json")))
cells = []
for name in idx["order"]:
    text = open(os.path.join(d, name)).read()
    src = text.splitlines(keepends=True)
    if name.endswith(".py"):
        cells.append(dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=src))
    else:
        cells.append(dict(cell_type="markdown", metadata={}, source=src))
nb = dict(cells=cells, metadata=idx["metadata"], nbformat=idx["nbformat"], nbformat_minor=idx["nbformat_minor"])
json.dump(nb, open(dst, "w"), indent=1, ensure_ascii=False)
print("wrote", dst, len(cells), "cells")
