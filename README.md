# Frugal_AI_Surrogate
This project implements a lightweight, fully-tuned surrogate model to predict CFD outcomes—Nu_avg and DeltaP—from U-bend geometry parameters. It enables rapid design evaluation, inverse design, and optimization without running expensive simulations.

## Highlights

- Predicts Nu_avg and DeltaP from 30 geometry parameters
- Benchmarked dropout, optimizer, and architecture with full traceability
- Feature importance analysis for design targeting
- Inverse design module to discover geometries that meet the target performance
- Modular scripts for training, sweeps, and summarization
- Ready for dashboard integration and Kubeflow automation

## Structure

- `configs/`: YAML config files for reproducible runs
- `data/`: Input dataset (CFD results)
- `models/`: Saved surrogate model
- `results/`: Organized outputs from sweeps and final runs
- `scripts/`: Core pipeline scripts (training, sweeps, inverse search)
- `notebooks/`: Demos and dashboard prototypes
- `utils/`: Helper modules for data loading, model building, metrics

##  How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Train and validate surrogate:
   ```bash
   python -m scripts.run_crossval
3. Sweep for different model parameters:
   ```bash
   python -m scripts.sweep_architecture
4. Summarize the results:
   ```bash
   python scripts/summarize_architecture.py
--> You can extend this pattern to other hyperparameters like activation, dropout and optimizer by creating corresponding scripts <--
```bash
python -m scripts.sweep_<hyperparameter>
python scripts/summarize_<hyperparameter>
```
## Results

- Best surrogate: depth=3, width=64, activation=leaky_relu, dropout=0.3
- R² Nu: 0.89+, RMSE Nu: ~10.85
- R² DP: 0.94+, RMSE DP: ~0.23
  
## Future developments

- Inverse design
- Surrogate-assisted sampling
- Kubeflow + Katib automation
- Dashboard integration - results visualization

## Data Source

This project uses CFD simulation data from the following publication:

Title: CFD dataset for U-bend geometries with varying design parameters
Authors: Mohamed A. El-Sayed, Ahmed M. Elshazly, Mohamed A. El-Gendy
Journal: Data in Brief, Volume 48, 2023
DOI: https://doi.org/10.1016/j.dib.2023.109477

Dataset: U_Bend_full_8933.csv
Description: Contains 8,933 random samples (from 10,000 samples) of U-bend geometries with corresponding CFD outputs (Nu_avg, DeltaP) across 30 parametric features, extracted and composed into a CSV file. 

  
## Author
Sajeed -- Aerospace Engineer.


## License
MIT


