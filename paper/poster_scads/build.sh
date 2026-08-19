#!/usr/bin/env bash
# Build the RankBind A0 poster.
#
# Runs probe.tex first: it reports how tall each column's content actually is
# against the space the page has for it, so a column that would run into the
# footer shows up as a number here instead of in print.
set -e
export PATH=/software/all/texlive/20230313-GCC-13.2.0/bin/x86_64-linux:$PATH
cd "$(dirname "$0")"

xelatex -interaction=nonstopmode probe.tex > probe.log 2>&1 || true
grep "RANKBIND-PROBE" probe.log | tail -5 | awk '
  { gsub(/RANKBIND-PROBE |pt/, ""); v[$1] = $3 }
  END {
    avail = v["colh-available"]
    printf "column fit (available %.1f cm):\n", avail * 0.03514
    for (c = 1; c <= 3; c++) {
      k = "col" c; h = v[k]
      printf "  %s  %5.1f cm  %s\n", k, h * 0.03514,
             (h <= avail ? "fits, " sprintf("%.1f cm spare", (avail-h)*0.03514) \
                         : sprintf("OVERFLOWS by %.1f cm", (h-avail)*0.03514))
    }
    printf "  band  %5.1f cm\n", v["band"] * 0.03514
  }'

for pass in 1 2; do
  xelatex -interaction=nonstopmode main.tex > "build_pass$pass.log" 2>&1 || true
done
echo "overfull boxes: $(grep -c Overfull build_pass2.log || echo 0)"
ls -la main.pdf
