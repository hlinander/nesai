import numpy
import json

N = 17
alpha = 0.25
dist = numpy.random.dirichlet([alpha] * N, 1000).tolist()
hfile = "#include <array>\n"
hfile += "static const std::array<std::array<float, 17>, 1000> dirichlet = {{ ";
pout = []
for d in dist:
    out = ""
    out += " { "
    out += ",".join(map(lambda p: "%.10f" % p, d))
    out += " } "
    out += "\n"
    pout.append(out)
hfile += ",".join(pout)
hfile += "}};"


with open("dirichlet.h", "w") as f:
    f.write(hfile)
