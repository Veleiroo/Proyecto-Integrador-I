# Setup for usage

For the correct execution of the code you may need to do the following:
   
1. Create a virtual environment (execute only once, otherwise packages will have to be reinstalled):
python -m venv .venv

2. Activate the virtual environment (you should see '(.venv)' appear in your terminal prompt):
.\.venv\Scripts\activate

3. Update pip:
python -m pip install --upgrade pip

4. Install the dependencies from the requirements file (this takes about 30' to finish the first time it is executed in the venv):
pip install -r requirements.txt

# Setup or development

For the correct execution of the code you may need to do the following:

1. Create a .env file and write your roboflow apikey there with this format: ROBOFLOW_API_KEY=your_api_key