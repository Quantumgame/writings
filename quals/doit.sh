#! /bin/sh

#rm *.aux *.bbl *.bcf *.blg *.log *.xml *.out *.toc
pdflatex quals && biber quals && pdflatex quals

# with biblatex, it is only necessary to run latex once after bibtex, not twice.
# with older tex installations, you may need to change 'biber' to 'bibtex'
