# Energy consumption forecast REST API
## Installation
* Install Python3+ (currently tested 3.5 on Ubuntu and 3.4 on Windows XP)
* Install virtualenv
```
pip install virtualenv
```
* Install Git Bash
```
cd PROJECT_FOLDER
```
* Create virtualenv
```
# deprecated
virtualenv .
# new and preferred way
python3 -m venv /path/to/new/virtual/environment
```
* Windows ONLY: activate virtual environment and manually install numpy and scipy through whl 
```
source Scripts/activate
pip install pip_windows/scipy-1.0.0-cp34-cp34m-win_amd64.whl
pip install pip_windows/numpy-1.13.3+mkl-cp34-cp34m-win_amd64.whl
manually run pip_windows/msodbcsql.msi
```
* Ubuntu/OSX ONLY:
```
sudo apt-get update
sudo apt-get install -y python-pip python-dev build-essential freetds-dev

source venv/bin/activate
pip install --no-cache-dir Cython
pip install numpy==1.14.3
pip install scipy==1.1.0
```
* All OS: Install other dependencies:
```
pip install -r requirements.txt
```

* Config SQL credentials in ENV variables or in credentials.json in the following format:
```
{

      "driver": "{ODBC Driver 13 for SQL Server}"
      "host": "server_ip",
      "database":"database_name",
      "user": "username",
      "password": "password",
      "calculate_historical_forecasts": true,
      "port": "1433"
}
```


## Test data
https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households