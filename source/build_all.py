import nbtools, importlib, sys
for mod, fn in [("nb_part_b", "Part_B_Audit_the_SDG_palette.ipynb"), ("nb_part_c", "Part_C_Ground_an_AI_palette.ipynb")]:
    m = importlib.import_module(mod)
    nbtools.write(m.cells, fn, f"./{fn}")
    print("wrote", fn, len(m.cells), "cells")

nbtools.make_module("./colour_masterclass.py")
print("wrote colour_masterclass.py")
