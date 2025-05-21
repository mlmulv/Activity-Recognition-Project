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
	<li><a href="#dependencies">Dependencies</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#authors">Authors</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project is a sequence classification project to identify if an individual is standing, walking, or running by analzying accelerometer data.

<!-- GETTING STARTED -->
## Getting Started

### Installation

*  Clone the repo
   ```sh
   git clone https://github.com/mlmulv/Activity-Recognition-Project.git

   ```

### Format 

Here is the format of the project files:

```
Activity-Recognition-Project/
├── Activity-Recognition-Project.ipynb
├── README.md
├── requirements.text
├── src/
│   └── preprocessing.py
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

This project was done using Anaconda for python packet managment. If you have not installed Anaconda, go to https://www.anaconda.com/download

* Create a conda environment with the *requirements.txt* file in the same directory where the git repo is cloned.
   ```sh
   conda create --name activityRecognition --file requirements.txt
   conda activate activityRecognition
   pip install hmmlearn==0.3.3

   ```
<!-- USAGE -->
### Usage

Within the *Acitivity-Recognition-Project.ipynb* file, I train and test Activity Recognition Classifier with the given data I collected. I use default values as defined within the src/ directory, but the values are module. The results of the classifier are displayed within the file.
	

<!-- Authors -->
## Authors

Markus Mulvihill
