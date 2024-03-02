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
virtualenv .
```
* Windows ONLY: activate virtual environment and manually install numpy and scipy through whl 
```
source Scripts/activate
pip install pip_windows/scipy-1.0.0-cp34-cp34m-win_amd64.whl
pip install pip_windows/numpy-1.13.3+mkl-cp34-cp34m-win_amd64.whl
```

* Ubuntu/OSX ONLY:
```
source venv/bin/activate
pip install numpy
pip install scipy
```
* All OS: Install other dependencies:
```
pip install -r requirements.txt
```

* Config SQL credentials in credentials.json

## Server
* forecasts are available through Falcon REST API. REST API must be served through WSGI server. We use waitress, because its cross-platform.
```
waitress-serve --listen=<LOCAL_IP>:<FREE_PORT> api.prediction_api:api
```

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