# TSKS15 – Detection and Estimation of Signals

Website and material for the [TSKS15 (Detection and Estimation of Signals)
course](https://studieinfo.liu.se/en/kurs/TSKS15/) at Linköping University,
Sweden.


## Website
The website with relevant information (schedule, labs, ...) can be found at
[klb2.github.io/tsks15/](https://klb2.github.io/tsks15/).


## Notebooks
Some topics of the course are illustrated by interactive Python notebooks.
They can be found in the `notebooks/` directory.

### Running Marimo Online
The easiest way to run the notebooks is by using the official
[marimo](https://marimo.app/) playground. This will run the notebooks in your
browser without the need to install anything locally.

Simply navigate to [https://marimo.app/](https://marimo.app/), click on `New
--> Open from URL...` and enter the URL to the notebook in this repository.
Alternatively, you can click on the links to the notebooks on the course
website at
[klb2.github.io/tsks15/notebooks.html](https://klb2.github.io/tsks15/notebooks.html).

### Local Installation
If you want to run the notebooks locally on your machine, Python3, marimo, and
some other libraries need to be installed.
The latest code was developed and tested with the following versions:

- Python 3.13.5
- numpy 2.3.2
- scipy 1.16.1
- matplotlib 3.10.3
- marimo 0.14.13

Make sure you have [Python3](https://www.python.org/downloads/) installed on
your computer.
You can then install the required packages by running
```bash
pip3 install -r notebooks/requirements.txt
```
This will install all the needed packages which are listed in the requirements 
file.

Finally, you can run the Marimo notebooks with
```bash
marimo run notebooks/<PATH_TO_NOTEBOOK.py>
```


## Contributing
You are welcome to contribute to this project.
You should be able to find all the information you need in [`CONTRIBUTING.md`](CONTRIBUTING.md).


## License
The content of this project itself is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International license](https://creativecommons.org/licenses/by-nc/4.0/), and the underlying source code used to format and display that content is licensed under the [MIT license](LICENSE).
