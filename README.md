<p align="center">
<h1 align="center"><strong>CRTM: Collective Robotic Terrain Modification</strong></h1>
  <p align="center">
    Collective Embodied Intelligence Lab
    <br>
    Cornell University 
  </p>

<div align="center">
	
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/Ericland/collective-terrain-modification-2d)
[![](https://img.shields.io/badge/DARS2024-%F0%9F%93%96-blue)](https://Ericland.github.io/files/papers/2024_DARS.pdf)

</div>

This repository contains the official 2D implementation accompanying our research on **collective terrain modification**.  
The work investigates how swarms of simple earthmoving agents can collectively reshape terrain from an **initial configuration** to a **target configuration** using distributed, local rules.

---

## 📑 Table of Contents

- [Research Context](#-research-context)
- [Contributions](#-contributions)
- [Repository Structure](#-repository-structure)
- [Requirements](#-requirements)
- [Running Experiments](#️-running-experiments)
- [Evaluation](#-evaluation)
- [Citation](#-citation)
- [Roadmap](#-roadmap)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 📖 Research Context

Collective construction and terrain modification are fundamental problems in swarm robotics and embodied intelligence.  
Our research explores:

- How large groups of simple robots can collaborate to perform complex modifications of 2D terrains.
- The tradeoffs between **centralized**, **single-robot**, and **distributed multi-robot** planning approaches.
- Scalable, fault-tolerant algorithms that leverage **local interactions** for global terrain transformations.

This repository provides the simulation environment, algorithms, and demonstration scripts used to validate our approach.

---

## 🧪 Contributions

This project enables:

- **Abstract terrain model**: Representing terrains as 2D height maps, with well-defined modification primitives.  
- **Planning algorithms**: Implementations for centralized, single-robot, and distributed multi-robot planning.  
- **Evaluation metrics**: Error to target terrain, number of actions, fault tolerance, and scalability.  
- **Demonstration scripts**: Reproducible experiments showing terrain merging, mound relocation, and random tasks.  
- **Visualization tools**: Plots and animations of terrain evolution during agent operation.

---

## 📂 Repository Structure

```
.
├── demo_merge_mounds.py     # demo: merge multiple mounds
├── demo_move_mound.py       # demo: relocate a mound
├── demo_random_task.py      # demo: random terrain task
├── src/                     # core modules (terrain, agents, strategies, metrics)
├── data/                    # example terrain data and parameters
├── image/                   # illustrative figures
├── video/                   # demo animations
├── LICENSE
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8+  
- `numpy`, `matplotlib`  
- (Optional) `scipy`, `tqdm`, or other helper libraries depending on experiment scripts  

To install dependencies:  
```bash
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

Run one of the demo scripts:

```bash
python demo_merge_mounds.py
python demo_move_mound.py
python demo_random_task.py
```

Modify parameters within each demo script to change terrain size, number of agents, or target configuration.  

---

## 📊 Evaluation

Our simulation framework supports the following analyses:

- **Error reduction curves** (distance to target terrain over time)  
- **Robustness to failures** (agent dropout, stochastic actions)  
- **Scalability** (increasing swarm size vs efficiency)  
- **Comparison across planning approaches**  

Researchers can reproduce plots from the paper or extend them with new strategies.

---

## 📚 Citation

If you use this codebase in your research, please cite:

```bibtex
@inproceedings{chen_2d_2024,
	title = {{2D} {Construction} {Planning} for {Swarms} of {Simple} {Earthmover} {Robots}},
	language = {en},
	booktitle = {Distributed {Autonomous} {Robotic} {Systems}},
	publisher = {Springer International Publishing},
	author = {Chen, Jiahe and Petersen, Kirstin},
	year = {2024},
}
```

---

## 🧭 Roadmap

- Extend to 3D terrain modification  
- Incorporate physical robot experiments  
- Explore reinforcement learning and adaptive distributed policies  
- Benchmark against additional planning baselines  

---

## 📄 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

This project is developed at the **Collective Embodied Intelligence Lab, Cornell University**.  
We thank our collaborators and funding sources
