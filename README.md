fpl-team-prediction-bot

# Fantasy Premier League Team Prediction Bot

This project is a Fantasy Premier League (FPL) team prediction and suggestion bot. It fetches data from the FPL API, processes player statistics, trains a machine learning model, and optimizes team selection under FPL rules. The project features a Flask web interface for easy team recommendations.

## Project Structure

```
fpl-team-prediction-bot/
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py      # Fetch data from the FPL API
│   ├── data_processor.py    # Process and prepare data for modeling
│   ├── model_trainer.py     # Train and load the machine learning model
│   └── team_optimizer.py    # Team selection logic under FPL rules
├── static/
│   └── style.css            # UI styling for the web interface
├── templates/
│   └── index.html           # Main HTML template for the Flask app
├── api_full.py              # Main Flask application (UI and logic)
├── requirements.txt         # Project dependencies
├── .gitignore               # Files and directories to ignore in version control
└── README.md                # Project documentation
```

## Setup

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd fpl-team-prediction-bot
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```sh
   python api_full.py
   ```


## Finding Your FPL ID

To use the bot for your own team, you will need your FPL (Fantasy Premier League) ID. Here's how to find it:

1. Log in to the [Fantasy Premier League website](https://fantasy.premierleague.com/).
2. Click on the "Pick team".
3. Click on the "Gameweek History"
4. Look at the URL in your browser's address bar. It will look like:

   ```
   https://fantasy.premierleague.com/entry/1234567/history
   ```

   Here, `1234567` is your FPL ID.
4. Use this number as your FPL ID in the api_full.py

---

## Usage

- The bot fetches player statistics and fixtures from the FPL API.
- It processes the data to create a dataset for the machine learning model.
- The model is used to predict expected points for each player.
- The team optimizer selects the best team of 15 players while adhering to FPL rules and budget constraints.
- The web UI displays the recommended team, captain, vice-captain, and key stats.

## Features

- Data fetching from the official FPL API
- Data processing and cleaning
- Machine learning model training and prediction
- Team optimization based on budget and squad structure
- Flask web interface for team recommendations

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or new features.