# CS 7641 Assignments

This repo is full of code for [CS 7641 - Machine Learning](https://www.omscs.gatech.edu/cs-7641-machine-learning) at Georgia Tech. 

A huge thanks to jontay ([https://github.com/JonathanTay](https://github.com/JonathanTay)) for sharing his code. Much of the code contained in this repo is based off of his work.

## Wait, code?

Yup, we are encouraged to steal code. All the code. It's fine. Only the analysis matters.

For more support of this claim, see [https://gist.github.com/cmaron/46f0992d42be87380c208086eec9797f](https://gist.github.com/cmaron/46f0992d42be87380c208086eec9797f)

For even _more_ support, we were asked to submit links to code rather than the actual code when submitting assignments partway through the semester.

## How do I use this?

If a python virtual environment has been setup for the project, a simple `pip install -r requirements.txt` should take care of the required packages.

Each assignment folder has its own `run_experiment.py` that will do most of the work for you. The big exception is assignment 2. Assignment 2, at least in Fall of 2018, was due soon after the midterm which was soon after the first assignment. These assignments take a while so I didn't put a ton of effort into doing anything fancy for assignment 2. Not to say any of this is fancy, obviously.

Running `python run_experiment.py -h` should provide a list of options for what you can do.

For the most part it is simple to run a given set of experiments based on a specific algorithm. One flag to consider always including is `--threads` with a value of `-1`. This will speed up execution in some cases but also might use all available cores.

The `--verbose` flag can be helpful to view data about a given dataset or MDP.

For assignments 3 and 4 plotting data is a separate step from generation. For those assignments the `--plot` flag should be used once data is generated

Each assignment folder should have its own readme with anything specific to not for that assignment.

## Why should I trust _you_, of all people?

Good question.

## ಠ_ಠ

Ok fine, I did _ok_ on the assignments (as in no grade less than 90). For the grades that were not 100 the feedback did not mention missing charts or values, so I'm confident this code does not miss anything major in that regard. 

That said, this is based off of the Fall 2018 semester. Things can change and you should always pay attention to announcements and go to office hours to be certain of the specifics. 

## But a thing is broken!?

Feel free to open an issue for things that are flat out broken (or even better open a PR) and I can take a look.

That said, caveat emptor applies. 
