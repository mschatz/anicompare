# anicompare

`anicompare` evaluates how sequence similarity estimated from `minimap2` alignments compares with similarity estimated from k-mer conservation using a Jaccard coefficient.

## Implementation Plan

The project is organized as a small pure-Python package with thin CLI wrappers. Shared library modules handle FASTA I/O, mutation modeling, k-mer and modimizer extraction, `minimap2` execution, SAM-based ANI estimation, resumable experiment orchestration, and plotting.

### Components

1. `generate-genome`
   - Generate a random genome of a requested length.
   - Write the result as a FASTA file.
   - Support deterministic output with a random seed.

2. `mutate-genome`
   - Read a FASTA genome from simulation or a real reference.
   - Apply configurable substitution, insertion, and deletion rates.
   - Record both requested and realized mutation counts in metadata.
   - Start with substitution-only experiments by setting insertion and deletion rates to zero.

3. `compute-jaccard`
   - Compare two genome FASTA files using canonical k-mers.
   - Default to `k=21`, but keep `k` configurable.
   - Support either exact k-mer sets or a sketching strategy based on modimizers.
   - Emit machine-readable output for downstream aggregation.

4. `run-experiment`
   - Accept either a real reference FASTA or a simulated genome length.
   - Simulate a reference genome first when length is supplied.
   - Analyze a configurable list of mutation rates.
   - Create a per-rate and per-replicate directory structure.
   - Place a mutated genome in each replicate directory and a symlink to the unmodified reference.
   - Run `minimap2`, parse SAM alignments, and estimate ANI from the SAM records.
   - Run Jaccard or modimizer comparison in the same directory.
   - Write per-run logs and aggregate a master TSV across all rates and replicates.

5. `plot-results`
   - Read the master TSV.
   - Produce scatterplots comparing true mutation rate, `minimap2` ANI, and Jaccard similarity.
   - Compute correlation statistics between these quantities.

### Project Layout

- `pyproject.toml`
- `src/anicompare/__init__.py`
- `src/anicompare/io.py`
- `src/anicompare/random_genome.py`
- `src/anicompare/mutate.py`
- `src/anicompare/kmers.py`
- `src/anicompare/modimizers.py`
- `src/anicompare/minimap2_ani.py`
- `src/anicompare/runner.py`
- `src/anicompare/plotting.py`
- `src/anicompare/cli_generate_genome.py`
- `src/anicompare/cli_mutate_genome.py`
- `src/anicompare/cli_jaccard.py`
- `src/anicompare/cli_run_experiment.py`
- `src/anicompare/cli_plot_results.py`

### Output Structure

The runner writes a resumable results tree:

- `results/run_config.json`
- `results/reference/reference.fa`
- `results/rate_<rate_label>/replicate_<n>/reference.fa`
- `results/rate_<rate_label>/replicate_<n>/mutated.fa`
- `results/rate_<rate_label>/replicate_<n>/mutation_metadata.json`
- `results/rate_<rate_label>/replicate_<n>/minimap2.sam`
- `results/rate_<rate_label>/replicate_<n>/minimap2_metrics.tsv`
- `results/rate_<rate_label>/replicate_<n>/jaccard_metrics.tsv`
- `results/rate_<rate_label>/replicate_<n>/run_summary.tsv`
- `results/rate_<rate_label>/replicate_<n>/run.log`
- `results/master_results.tsv`
- `results/plots/`

### Restartability

The experiment runner stores the full configuration in `run_config.json`. If the output directory already exists, the runner loads the configuration and:

- skips work when the expected outputs already exist,
- resumes incomplete runs,
- rejects incompatible parameter changes unless explicitly forced.

This makes it easy to restart long experiments and add missing outputs without rerunning completed work.

### Parallelism

- `minimap2` thread count is configurable independently with `--minimap2-threads`.
- Python-side parallelism is controlled with `--workers` so rate/replicate jobs can run concurrently.
- The implementation separates concurrent jobs from per-job `minimap2` threads to avoid oversubscribing CPUs.

### Dependencies

The code uses the Python standard library as much as possible. The only required external executable is `minimap2`. Plotting uses `matplotlib` if available.

## Planned Command-Line Interfaces

### Generate a random genome

```bash
python -m anicompare.cli_generate_genome \
  --length 5000000 \
  --output reference.fa \
  --seed 7
```

### Mutate a genome

```bash
python -m anicompare.cli_mutate_genome \
  --input-fasta reference.fa \
  --output mutated.fa \
  --substitution-rate 0.01 \
  --insertion-rate 0.0 \
  --deletion-rate 0.0 \
  --seed 11
```

### Compute exact k-mer Jaccard

```bash
python -m anicompare.cli_jaccard \
  --fasta-a reference.fa \
  --fasta-b mutated.fa \
  --k 21
```

### Run the full experiment

```bash
python -m anicompare.cli_run_experiment \
  --simulate-length 5000000 \
  --output-dir results \
  --replicates 3 \
  --mutation-rates 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.10 0.15 \
  --substitution-scale 1.0 \
  --insertion-scale 0.0 \
  --deletion-scale 0.0 \
  --k 21 \
  --workers 3 \
  --minimap2-threads 4 \
  --seed 17
```

### Plot results

```bash
python -m anicompare.cli_plot_results \
  --input results/master_results.tsv \
  --output-dir results/plots
```
