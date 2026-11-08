# The Gemini theme uses fontspec, so the poster must be built with LuaLaTeX.
# Overriding $pdflatex also covers editors that call `latexmk -pdf`.
$pdflatex = 'lualatex %O %S';
