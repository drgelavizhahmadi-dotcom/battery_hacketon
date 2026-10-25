# paper2/tex/Makefile: two builds of the same sources.
#   make            -> anonymous.pdf and preprint.pdf
#   make check      -> search both PDFs (text and metadata) for the author name and repository owner
#   make figures    -> rebuild paper2/figures (pinned interpreter, see PYTHON)
#   make review     -> REVIEW_ITEMS.md
#   make submission -> submission_tmlr/ with exactly what goes to TMLR, verified anonymous
#   make clean
TEXBIN := $(HOME)/Library/TinyTeX/bin/universal-darwin
# Pinned interpreter: the system Python 3.9.6 that produced every committed result (numpy 1.26.4, matplotlib 3.9.4,
# torch 2.8.0, scipy 1.13.1, pandas 2.3.3). Do not rely on `python3` from PATH.
PYTHON := /usr/bin/python3
LATEXMK = PATH="$(TEXBIN):$$PATH" latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error
GEN = title.tex body.tex $(foreach x,A B C D E,appendix$(x).tex appendix$(x)_anon.tex)
SRC = main.tex references.bib tmlr.sty tmlr.bst fancyhdr.sty $(GEN)

all: anonymous.pdf preprint.pdf

$(GEN): build_tex.py ../draft/paper2_draft.md ../figures/MANIFEST.md
	cd ../.. && $(PYTHON) paper2/tex/build_tex.py

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

figures:
	cd ../.. && for f in make_fig1 make_fig2 make_fig3 make_fig4 make_fig5 make_fig6 make_fig7 make_fig8 make_fig9 make_supp; do $(PYTHON) paper2/figures/$$f.py || exit 1; done

review: $(GEN)
	cd ../.. && $(PYTHON) paper2/tex/gen_review_items.py > /dev/null

submission: anonymous.pdf
	rm -rf submission_tmlr && mkdir submission_tmlr
	cp anonymous.pdf submission_tmlr/
	# later: anonymised code archive -> submission_tmlr/code_anonymised.zip (not built yet)
	$(PYTHON) check_submission.py submission_tmlr

.PHONY: all check clean figures review submission
