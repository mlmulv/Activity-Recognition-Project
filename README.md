<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
	<li><a href="#format">Format</a></li>
	<li><a href="#dependencies">Dependencies</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#usage">Completion</a></li>
    <li><a href="#authors">Authors</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project is a sequence classification project to identify if an individual is standing, walking, or running by analzying accelerometer data.

<!-- GETTING STARTED -->
## Getting Started

### Installation

*  Clone the repo and move into the directory
   ```sh
   git clone https://github.com/mlmulv/Activity-Recognition-Project.git
   cd Activity-Recognition-Project
   ```
<!-- FORMAT -->
### Format 

Here is the format of the project files:

```
Activity-Recognition-Project/
├── Activity-Recognition-Project.ipynb
├── README.md
├── environment.yml
├── .gitignore
├── Activity-Recognition-Report.pdf
├── src/
│   ├── preprocessing.py
│   └── HMM.py
└── data/
    ├── run/
    │   ├── run_1.csv
    │   └── run_2.csv
    ├── stand/
    │   ├── stand_1.csv
    │   └── stand_2.csv
    └── walk/
        ├── walk_1.csv
        └── walk_2.csv
```

### Dependencies

This project was done using Anaconda for python packet management. If you have not installed Anaconda, go to https://www.anaconda.com/download

Run the following lines of code below within the Anaconda Prompt or Command Prompt if you have Anaconda configured on it.

* Create a conda environment with the *environment.yml* file within the Activity-Recognition-Project directory. *If you do not follow this step, conda won't be able to find the enviornment.yml file*.
   ```sh
   conda env create -f environment.yml
   conda activate activityRecognition
   ```
* Open a notebook
   ```sh
  python -m notebook

   ```
   
<!-- USAGE -->
### Usage

Within the *Acitivity-Recognition-Project.ipynb* file, I conduct data preprocessing, data visualization, training our HMM, and testing our trained modal. I use default parameters as defined within the src/ directory, but the parameters are modular. The results of the classifier are displayed within the file.

The functions and classes used in are documented in *Acitivity-Recognition-Project.ipynb* are documented accordingly in the *src/*.
	
<!-- COMPLETION -->
### Completion

Once you are done tinkering and analyzing the code. You can remove the conda environment if you wish to save space on your system.

```sh
conda deactivate
conda remove --name activityRecognition --all   
```

<!-- Authors -->
## Authors

Markus Mulvihill
