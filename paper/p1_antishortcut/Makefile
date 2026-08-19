PAPER  = main
LATEX  = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex
.PHONY: all clean
all: $(PAPER).pdf
$(PAPER).pdf: $(PAPER).tex
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)
clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).toc $(PAPER).bbl $(PAPER).blg
