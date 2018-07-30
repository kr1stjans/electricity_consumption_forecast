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

## Packaging

Before packaging make sure to create a new tag. Use <Major>.<Minor> versioning. 
Bump minor version for bug fixes or non-interface related changes.
Bump major version for major refactor or API interface changes or improvements.

After creating the tag update changelog. The easiest way to do so is via git changelog (see package git-extras)
Remove any non-informative entries from the changelog.
```
git changelog
```

To package the API:
```
cd ..
. ./flexibility-forecast-api/package.sh
```

A zip of the form flexibility-forecast-api-<TAG>.zip will be created with temporary files removed.

## Deployment (Windows)

* stop Windows service that controls the server
* remove .pyc files from server directory
* unzip and copy over files from package flexibility-forecast-api-<TAG>.zip (by default c:\slutils\drcs-api)
* start Windows service

## Server
* forecasts are available through Falcon REST API. REST API must be served through WSGI server. We use waitress, because its cross-platform.
```
waitress-serve --listen=<LOCAL_IP>:<FREE_PORT> api.prediction_api:api
```

## Windows System Service

* Make sure to install NSSM http://nssm.cc/download

* Create a run\_forecast\_api.bat file with the following contents
```
cmd /k "cd /d C:\slutils\drcs-api\ & .\Scripts\activate & waitress-serve --listen=10.243.120.3:8889 api.prediction_api:api"
```
* Run and set path to created .bat file in shown UI
```
nssm install Forecast API Server
```
* Start the service.


## API
* \<IP\>:\<PORT\>/newData accepts the following JSON format:
```
[{
    'dataSourceValuesGroupedByDate': {<DS_ID>: {<DATE>: <VALUE>}},
    'source' : <SOURCE_URL>
}]
```
* Received data is saved (created or updated) in local SQL database. Forecast module is called at the end to calculate forecast based on the new received data
* New forecast is saved to database and sent as POST request to subscriber in the same JSON format
