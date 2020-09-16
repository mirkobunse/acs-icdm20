all: \
  plots/kl/kl_main.pdf \
  plots/proportion_impact/proportion_impact_main.pdf \
  plots/proportion_impact/proportion_impact_dt.pdf \
  plots/proportion_impact/proportion_impact_svc.pdf

# compile PDFs from plot CSVs
plots/kl/kl_main.pdf: plots/kl/kl_main.tex plots/kl/main.csv
	- pdflatex -halt-on-error -interaction=nonstopmode --output-directory plots/kl plots/kl/kl_main.tex
define PDF_RULE
plots/proportion_impact/proportion_impact_$T.pdf: plots/proportion_impact/proportion_impact_$T.tex plots/proportion_impact/$T_ProportionalLabelScore.csv
	- pdflatex -halt-on-error -interaction=nonstopmode --output-directory plots/proportion_impact plots/proportion_impact/proportion_impact_$T.tex
endef # this rule is evaluated for each *.pdf target of proportion_impact
$(foreach T,main dt svc,$(eval $(PDF_RULE)))

# generate plot CSVs (aggregations) from results
plots/kl/%.csv: sandbox/plots/kl.py
plots/%.csv: results/%.csv
	python -m sandbox.plots $< $@
define PLOT_RULE
plots/proportion_impact/$T_ProportionalLabelScore.csv: results/proportion_impact/$T.csv results/acs/$T.csv sandbox/plots/proportion_impact.py
	python -m sandbox.plots --additional results/acs/$T.csv --scorer ProportionalLabelScore results/proportion_impact/$T.csv plots/proportion_impact/$T_ProportionalLabelScore.csv
endef
$(foreach T,main dt svc,$(eval $(PLOT_RULE)))

# conduct experiments
results/dataset_info/%.csv: sandbox/experiments/dataset_info.py
results/gridsearch/%.csv: sandbox/experiments/gridsearch.py
results/kl/%.csv: sandbox/experiments/kl.py
results/proportion_impact/%.csv: sandbox/experiments/proportion_impact.py
results/acs/%.csv: sandbox/experiments/acs.py
results/%.csv: config/experiments/%.yml
	python -m sandbox.experiments $< $@

# perform parameter-tuning
GRIDSEARCH_CONFIGS = \
  config/experiments/gridsearch/main.yml \
  config/experiments/gridsearch/dt.yml \
  config/experiments/gridsearch/svc.yml
gridsearch: $(patsubst config/experiments/%.yml,results/%.csv,$(GRIDSEARCH_CONFIGS))

# pytest runs all scripts in tests/*.py
tests:
	pytest tests/__init__.py

clean:
	rm -r $(foreach T,aux log pdf csv,$(shell find plots -name "*.$T")) $(shell find results -name "*.csv")

.PHONY: all gridsearch tests clean
.SECONDARY:
