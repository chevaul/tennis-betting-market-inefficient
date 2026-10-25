# tennis_betting_efficiency

GitHub repository for LSESU Data Science Society AT Project Showcase project "Predictive Modeling and Market Efficiency in Sports Betting."

## Overview

This project explores predictive modeling and market efficiency in tennis betting. We analyze betting odds data to:
- Assess market efficiency by comparing implied probabilities from odds to actual match outcomes
- Build predictive models for match outcomes using player statistics and historical data
- Evaluate calibration of betting markets across different surfaces, ranks, and tournaments
- Identify potential betting opportunities where markets may be inefficient

## Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/eaglemascot/tennis_betting_efficiency.git
cd tennis_betting_efficiency
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt lists package names without version constraints. For production use, consider pinning specific versions for reproducibility:
```bash
pip freeze > requirements-lock.txt
```

4. Add your tennis data files to the `data/raw/` directory (CSV format)

## Project Structure

```
tennis_betting_efficiency/
├── data/
│   ├── raw/          # Raw data files (CSV)
│   └── processed/    # Cleaned and processed data
├── docs/             # Documentation
├── notebooks/        # Jupyter notebooks for exploration
│   └── EDA.ipynb     # Exploratory data analysis
├── src/              # Source code
│   ├── features.py   # Feature engineering functions
│   ├── models.py     # Modeling utilities
│   └── utils.py      # Helper functions
├── requirements.txt  # Python dependencies
├── LICENSE           # MIT License
└── README.md         # This file
```

## Workflow

1. **Data Collection**: Place raw tennis match and betting odds data in `data/raw/`
2. **Exploratory Analysis**: Use `notebooks/EDA.ipynb` to explore data patterns, compute implied probabilities, and assess market calibration
3. **Feature Engineering**: Use functions in `src/features.py` to create predictive features
4. **Model Development**: Build and evaluate models using `src/models.py`
5. **Analysis**: Document findings in `docs/` and create visualizations

## Usage

### Running Exploratory Data Analysis

```bash
jupyter notebook notebooks/EDA.ipynb
```

The EDA notebook includes:
- Data loading and cleaning
- Computing implied probabilities from betting odds
- Analyzing missing data
- Visualizing player rank and surface distributions
- Creating calibration plots to assess market efficiency

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
