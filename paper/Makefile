PAPER  = main
BIB    = references
LATEX  = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(BIB).bib figures/*.png
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

view: $(PAPER).pdf
	xdg-open $(PAPER).pdf >/dev/null 2>&1 &

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).toc \
	      $(PAPER).bbl $(PAPER).blg $(PAPER).synctex.gz \
	      $(PAPER).fls $(PAPER).fdb_latexmk

distclean: clean
	rm -f $(PAPER).pdf
