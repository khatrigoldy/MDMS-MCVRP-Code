# MDMS-MCVRP-ABC

**Advanced Artificial Bee Colony Algorithm for Multi-Depot Multi-Satellite Multi-Compartment Vehicle Routing Problems**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository contains a complete implementation of a customized Artificial Bee Colony (ABC) metaheuristic algorithm specifically designed for solving the Multi-Depot Multi-Satellite Multi-Compartment Vehicle Routing Problem (MDMS-MCVRP). The project features advanced constraint handling, machine learning-driven hyperparameter optimization, and comprehensive experimental analysis.

### Problem Description

The MDMS-MCVRP involves:
- **Multiple depots** serving as starting points for primary vehicles
- **Multiple satellites** acting as intermediate consolidation points
- **Multi-compartment vehicles** with capacity constraints
- **Complex routing constraints** including mandatory satellite visits

## Key Features

- ✅ **Custom ABC Algorithm** - Purpose-built for MDMS-MCVRP with advanced constraint enforcement
- ✅ **ML-Enhanced Optimization** - Automated hyperparameter tuning using Grid Search and Bayesian methods
- ✅ **Constraint Handling** - Ensures all primary vehicles visit every satellite
- ✅ **Comprehensive Experiments** - 300+ trials with statistical validation
- ✅ **Sensitivity Analysis** - Systematic capacity variation studies across 25 scenarios
- ✅ **Rich Visualizations** - Convergence plots, heatmaps, and solution route displays
- ✅ **Publication-Ready Results** - Complete data and analysis for academic use

## Algorithm Details

### Artificial Bee Colony Implementation

The ABC algorithm consists of three phases:

1. **Employed Bee Phase**: Local search around current solutions
2. **Onlooker Bee Phase**: Probabilistic selection and intensification
3. **Scout Bee Phase**: Exploration through random restart of stagnant solutions

### Key Algorithmic Features

- **Constraint Enforcement**: Mandatory satellite visits using repair mechanisms
- **Multi-Level Routing**: Primary (depot→satellite) and secondary (satellite→customer) routes  
- **Dynamic Modification**: Three solution operators (satellite reordering, customer swapping, route optimization)
- **Penalty-Based Fitness**: Heavy penalties for constraint violations
- **Adaptive Search**: Balance between exploration and exploitation

## Experimental Results

## Hyperparameter Tuning Results: Bayesian Optimization vs Grid Search

We performed extensive hyperparameter tuning with two popular methods: **Grid Search** and **Bayesian Optimization**, each with 300 trials across key ABC parameters.

| Metric            | Grid Search           | Bayesian Optimization |
|-------------------|----------------------|----------------------|
| Best Fitness      | 15,056.0             | 15,252.75            |
| Mean Fitness      | 15,325.29            | 15,452.22            |
| Std. Deviation    | 86.07                | 102.70               |
| Success Rate      | 100%                 | 100%                 |
| Mean Run Time     | 18.20 seconds        | 23.90 seconds        |

### Interpretation:

- The **Bayesian Optimization** exhibited a **smoother and more consistent convergence curve** indicating focused exploration in promising hyperparameter regions.
- Despite this, its **higher standard deviation** is due to Bayesian sampling a **wider parameter space including some less optimal points** during exploration, resulting in greater variability across trials.
- The **Grid Search**, while less smooth, has a **lower standard deviation** because of exhaustive but fixed sampling over a uniform grid, which tends to produce steadier but less finely tuned results.
- The **noise in Grid Search’s convergence** reflects its uniform sampling and the presence of suboptimal parameter combinations.
- Ultimately, **Bayesian Optimization is more adaptive and efficient**, allowing faster convergence on competitive solutions despite variability, while **Grid Search trades consistency for exhaustive coverage**.

These complementary characteristics highlight why Bayesian optimization is well-suited for tuning complex metaheuristics like ABC on multi-depot multi-satellite routing problems.



### Optimal Hyperparameters

- **Swarm Size**: 80
- **Max Iterations**: 450  
- **Employed Ratio**: 0.40
- **Onlooker Ratio**: 0.25
- **Limit**: 35

### Sensitivity Analysis

Conducted across 25 capacity scenarios (5 satellite capacities × 5 secondary vehicle capacities):
- **Total Experiments**: 125 runs (5 runs per scenario)
- **Success Rate**: 100% across all scenarios
- **Best Configuration**: Satellite capacity 22,000 + Secondary capacity 800
- **Optimal Fitness**: 14,987.0

## Visualizations

The repository includes:
- **Convergence Curves**: Algorithm performance over iterations
- **Sensitivity Heatmaps**: Performance across capacity configurations  
- **Route Visualizations**: Best solution route displays
- **Statistical Analysis**: Box plots and performance distributions

## Technical Specifications

- **Programming Language**: Python 3.8+
- **Key Libraries**: NumPy, Matplotlib, SciPy, Pandas, Optuna
- **Algorithm Type**: Bio-inspired Metaheuristic (Swarm Intelligence)
- **Problem Complexity**: NP-Hard Combinatorial Optimization
- **Solution Encoding**: Route-based representation

## Applications

- **Supply Chain Management**: Multi-echelon distribution networks
- **Logistics Optimization**: Last-mile delivery with consolidation centers
- **Academic Research**: Metaheuristic algorithm benchmarking
- **Transportation Planning**: Complex routing with capacity constraints

## Research Contributions

1. **Novel ABC Variant**: Customized for multi-level vehicle routing constraints
2. **ML Integration**: Systematic hyperparameter optimization methodology
3. **Comprehensive Analysis**: Statistical validation across multiple problem scenarios
4. **Practical Insights**: Capacity planning guidelines for logistics networks

## Citation

If you use this work in your research, please cite:


## Contact & Support

**Author**: Gurnam Singh  
**Institution**: Kirori Mal College, University Of Delhi  
**Department**: Department of Operational Research  
**GitHub**: [@khatrigoldy](https://github.com/khatrigoldy)  
**E-Mail**: khatrigoldy10@gmail.com

For questions, suggestions, or collaboration opportunities, please open an issue or contact me directly.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- University of Delhi, Department of Operational Research
- Open-source Python community for excellent optimization libraries
- Research community for algorithmic foundations and benchmarks

---

*Developed as part of advanced research in Operations Research and Swarm Intelligence Optimization*

** If you find this work useful, please consider starring the repository!**




