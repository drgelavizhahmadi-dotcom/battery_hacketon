# paper2/tex/Makefile: two builds of the same sources.
#   make            -> anonymous.pdf and preprint.pdf
#   make check      -> search both PDFs (text and metadata) for the author name and repository owner
#   make clean
TEXBIN := $(HOME)/Library/TinyTeX/bin/universal-darwin
LATEXMK = PATH="$(TEXBIN):$$PATH" latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error
GEN = title.tex body.tex $(foreach x,A B C D E,appendix$(x).tex appendix$(x)_anon.tex)
SRC = main.tex references.bib tmlr.sty tmlr.bst fancyhdr.sty $(GEN)

all: anonymous.pdf preprint.pdf

$(GEN): build_tex.py ../draft/paper2_draft.md ../figures/MANIFEST.md
	cd ../.. && python3 paper2/tex/build_tex.py

anonymous.pdf: anonymous.tex $(SRC)
	$(LATEXMK) anonymous.tex

preprint.pdf: preprint.tex $(SRC)
	$(LATEXMK) preprint.tex

check: anonymous.pdf preprint.pdf
	@for f in anonymous preprint; do \
	  echo "== $$f.pdf"; \
	  pdftotext $$f.pdf - | grep -i -c -E "ahmadi|gelavizh|drgelavizhahmadi|github.com/drg" || true; \
	  pdfinfo $$f.pdf | grep -i -E "author|title|creator|producer|subject|keywords"; \
	  strings $$f.pdf | grep -i -c -E "ahmadi|gelavizh|drgelavizhahmadi" || true; \
	done

clean:
	PATH="$(TEXBIN):$$PATH" latexmk -C anonymous.tex preprint.tex

.PHONY: all check clean
