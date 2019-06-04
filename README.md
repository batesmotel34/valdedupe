# valdedupe

Validity Take home exercise for Robin Burr

## Installation and dependencies
This assumes that Python V3.7 or later is installed.

### Linux and macOS
After cloning the repo execute this command in the top level directory of the cloned repo to complete set up:
```
python3 -m venv env  && source env/bin/activate && pip install -r requirements.txt
```

## Getting Started

### Web App
```
flask run
```
### Command line for training

The dedupe library allows the user to perform training for the app by providing possible duplicates to which the user can provide a positive or negative response. This exercise includes the settings_file from training so the web app version uses that. To perform additional training, the app can be run as a command from the command line:
```
flask valdupe_train data/advanced.csv
```

