<h2>Automated Sudoku Solver</h2>

<p>This repo contains all the .ipynb notebooks and .py files for an automated sudoku solver, along with the LeNET model trained and its weights in hdf5 format.</p>

<h3>Tech Stack:</h3>
<li>Python, PIP, Streamlit</li>
<li>OpenCV, Tensorflow, Sklearn</li>
<li>OS, PIL, etc. auxilliary libraries</li>

<h3>Working:</h3>
<p>This project aims to identify uploaded sudoku boards from the images and capture the board position irrespective of warping and colour disparities and noise. Warping is handled by a warp transformation matrix that maps the 4-points of the contour of the board (largest area closed polygon) to a rectangle of defined dimesnions.</p>
<p>Other image processing like Grayscaling, Blurring and Cntour Detection are used to remove noise as much as possible.</p>
<p>LeNET CNN was trained from scratch on <b>Printed digits</b> dataset to clearly identify the digits from the cells and perform backtracking algorithm to determine the solution.</p>

<h5>Note: This still faces challenges in low light and bad distortion / camera noise situations. Collaboration is highly aprreciated.</h5>
