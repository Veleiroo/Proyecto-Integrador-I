# Setup

For the correct execution of the code you may need to do the following:

1. Create a .env file and write your roboflow apikey there with this format: ROBOFLOW_API_KEY=your_api_key
   
2. Create a virtual environment:
python -m venv .venv

3. Activate the virtual environment (you should see '(.venv)' appear in your terminal prompt):
.\.venv\Scripts\activate

4. Update pip:
python -m pip install --upgrade pip

5. Install the dependencies from the requirements file (this takes about 30' to finish the first time it is executed in the venv):
pip install -r requirements.txt

# Baseline

We are going to use as a baseline a pretrained model. The system that uses it is in baselines\roboflow. It was downloaded from roboflow:posture_correction_v4.

# sideView

This folder contains the code which we used to determine the best pose estimation model for the side view images.
