**Please read through the entire Jupyter notebook for this Lab Assignment before you start the assignment.**

## Problem Statement
A truly random game of rock-paper-scissors would result in a statistical tie with each player winning, tying, and losing one-third of the time. However, people are not truly random and thus can be studied and analyzed. You can play rock-paper-scissors against a computer   either [here](https://www.afiniti.com/corporate/rock-paper-scissors) or [here](http://www.essentially.net/rsp/play.jsp) or [here](https://www.cs.stir.ac.uk/~kms/schools/rps/index.php).

While the computer won’t win all rounds, over time it can exploit a person’s tendencies and patterns to gain an advantage over its opponent. 

As rock-paper-scissors enthusiasts and budding machine learning engineers, your team is tasked to build a rock-paper-scissors model/algorithm that can learn the opposing computer's tendencies/patterns and out-maneuver it. 

Can your model beat the computer?

## Prerequisits
* Python installation (Miniconda **highly recommended** - follow instruction [here](https://docs.anaconda.com/miniconda/install/#quick-command-line-install))
    * If using Miniconda:
        * Setup the environment using **Python 3.12** `conda create -n engs106 python=3.12`
        * Active the environment `conda activate engs106`
* (Recommended) Visual Studio Code ([Install here](https://code.visualstudio.com/download))
    * (If VSCode) Required extensions (left panel in VSCode): Python and Jupyter
    * If using Miniconda and VSCode, make sure to link Miniconda to VSCode
        * `Ctrl+Shift+P`
        * Choose the path to your environment
        * Make sure you can run the required cells
* (REQUIRED) Python packages: numpy, ipykernel, jupyter
    * If using conda:
        * `conda install numpy`
        * `conda install jupyter`
        * `conda install ipykernel`
    * If using pip (no miniconda; standard Python install):
        * `pip install numpy`
        * `pip install jupyter`
        * `pip install ipykernel`


### Design on Paper
A design on paper comprises the analysis, diagrams, pseudo-code, plans, and documents that present a complete and thorough solution to the task at hand. 

>[!CAUTION]
>**You never want to code or build a complex system without first creating a design on paper.** Our code should only ever be a translation of our solution into an implementation. 

Developing a design on paper serves three key purposes:
1. Developing your design will help you organize and crystalize your solution. Our goal is to achieve clarity of thought through a design on paper.
2. Your design on paper will serve as a ground truth as designs get larger and span months. Over time and as projects grow in scope and complexity, we might forget key details in our design (think of your working memory as RAM and your paper design as information stored on a disk). This is particularly important as we debug our systems, as we cannot effectively debug unless we know *exactly* what our outputs are supposed to look like. How can you say that something is fixed if you aren't totally sure how it is supposed to work?
3. Your paper design enables you to communicate clearly with your teammates and, eventually, with people who are not familiar with the intricate details of your design (like the TA that is grading your lab).
   
This is not just scratch paper--this is a formal solution to the problem that could be used to communicate your solution to others (indeed, you will need to communicate your solution to your course instructors for a grade, shortly). You want it to be concise, clear, and well organized. 

Your code should always be a translation of a fully-formed solution to a problem (read as: a design) into an implementation. If you are trying to solve the task at hand while programming, you need to revisit your design on paper. 

Work with your teammates to plan your strategy for solving this problem. 

### Generating your Training Data
Play the game of the rock-paper-scissors against the computer **at least 100 times** while you record the outcome (your moves & the computer’s move) at each time into a text (`.txt`) ﬁle. Record the score at the bottom of the ﬁle, i.e. your number of wins, losses and ties out of N tries (N ≥ 100). You will submit this data as part of your lab. Think about how you want to generate your training data. Is it better for each person to generate a little training data? How might that be different than one person playing? Remember, the computer is going to be learning from you. If it learns from multiple people is that better or worse than it learning from one person with more data available to their model?

### Implement your Algorithm
Now, leverage this data train your algorithm. Have your algorithm take as an input the game history (what was played by each team, what was the outcome) and estimate your best move for the next round. Implement your algorithm with Python in the notebook `assignment-1.ipynb`.

### Test Your Algorithm
Play the computer again at least the same number of times you played before, but this time using your program. Record the score (hopefully there is an improvement in your score!). You may use your program manually with the computer by entering in the computer response at each round to your program, or for an extra challenge (but no extra points) you can write a script to interact with the webpage (this goes beyond the scope of what you have been taught in this course and is not expected). Record data for each game, just like you did when you created your training data. Please submit this model evaluation data as part of your lab.

>[!TIP]
>The more you play the better you may be at guessing what the computer is up to. You can also develop your playing system using data from one website and see if you can use it to improve your performance on the other website (transfer learning and generalization), though this is optional.

## Roadmap
In this lab, you should:
1. Work through the `assignment-1.ipynb` notebook.
2. Collaborate with your team members to design on paper (plan out with diagrams, figures, etc.) your own Rock, Paper, Scissors model/algorithm.
3. Generate training data by playing (as a human) against one of the aforementioned online models.
4. Implement and train your model/algorithm in `assignment-1.ipynb`.
5. Then, test your algorithm against the computer.

## Goals of this Lab
There are three primary goals in this lab assignment:

1. To collaborate with your team members to approach an open-ended design problem. 

2. To consider the steps you need to take to develop and evaluate a predictive model. You must justify your design decisions. 

3. To have fun! And to get to know your team members better. 

## Expectations
### Labs, Generally
All lab submissions are individual and every item you submit should be a reflection of your own work. You should have ownership over the entirety of any lab you submit in this course. While your work is your own, we understand that it can be helpful in learning machine learning to collaborate with your peers, which can range from high-level discussion of a problem to debugging. Having others look at our code encourages us to write code with readability in mind. In practice, we will never work in a silo, and being able to discuss these topics with others well is a valuable skill. When you collaborate with another student, please cite them appropriately and be respectful of sharing too much. The Academic Honor Principle applies.

We respectfully ask that in the interest in furthering your own understanding of the material, that you refrain from using generative AI to code for you. Your work should be your own and you should feel comfortable justifying each design decision you make. 

Please cite any outside sources you reference.

### This Lab, Specifically
Though you can and should work closely on this assignment with your team members (split up generating the training data, developing your python program, etc.) and you will likely have very similar submissions (co-developed model/algorithm/code, etc), this is still an individual submission and you should have ownership over everything you submit. You should feel comfortable explaining how your system works to others (this is a great way to learn too--tell your friends and families about your brand new rock-paper-scissors machine!). Your written after-the-lab analysis should be your own. Each person should maintain and submit their own GitHub repo for this assignment. 

## Evaluation
This is an assignment that will be submitted for evaluation. 

At the end of the lab, you should add your Lab TA as a collaborator to your GitHub repository (remember, this is an individual submission, not a team submission). Please do not share this link with other students outside your team. 

You will be evaluated on your ability to communicate your ideas in this lab clearly and succinctly. This includes both the written component, where you describe your system, and any code you write. Comment and document as you go. Do not try to retrofit your documentation! (Though as you work the problem you might want to change earlier comments--this is ok and encouraged as needed!).

**You will not be penalized for failing to outperform the computer.** You will not be evaluated on the performance of your model in this lab. Rather, you will be evaluated on the quality of your planning, documentation, and analysis. Focus on communicating clearly and you will do well on this assignment. After all, not all great ideas work as well as we would hope! This is not reason to disregard those ideas--we should seek to learn from them. Sometimes a model that performs poorly will teach us more than a model that performs well. We should not be adverse to failure, but instead strive to understand failure modes.

We have not covered any ML techniques yet in the class yet, so this is just a warm up problem. Just do the best you can! This is a space to be creative--which leads us to the last, most important thing to keep in mind as you work through this lab:

>[!Important]
>Have fun with this assignment!

### What to Submit

**Note: This is the first assignment, designed to help you get familiar with the GitHub workflow.**

**For this assignment only**, please submit a `.zip` file containing:
1. Your training data, included as a `training.txt` file.
2. Your model evaluation data, generated in your Jupyter notebook.
3. A brief write-up describing your algorithm in a couple of sentences, including figures if necessary (your design on paper).
4. Finally, please individually briefly reflect on your experience in a couple of sentences. What are you taking away from this lab?

Your write-up detailing your design on paper and your reflection can be submitted as a single document, either as a markdown (`.md`) file or as a PDF.

**Starting from Assignment 2**: Next week, we will release a Google Sheet assigning each student a Lab TA. From Assignment 2 onwards, you will need to:
- Submit a link to your GitHub repository (instead of a .zip file)
- Add your assigned Lab TA as a collaborator to your repository, or they won't be able to grade your work!
- When you are done with the lab, push all your work to your personal remote repository and check that everything is up-to-date using a web browser. The web-view of your remote repository is what will ultimately be evaluated--not the work in your local environment!
