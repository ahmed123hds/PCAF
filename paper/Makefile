TEX = pcaf_paper

.PHONY: all clean

all:
	pdflatex $(TEX).tex
	bibtex $(TEX)
	pdflatex $(TEX).tex
	pdflatex $(TEX).tex

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).pdf
