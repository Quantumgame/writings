#! /bin/sh

rm *.bak *.aux *.dvi *.lof *.bbl *.bcf *.log *.lot *.xml

latex thesis && \
biber thesis && \
latex thesis && \
pdflatex thesis.tex

# with biblatex, it is only necessary to run latex once after bibtex, not twice.
# with older tex installations, you may need to change 'biber' to 'bibtex'
