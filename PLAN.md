1. Define the problem
Before touching the data, write down:
- What am I trying to predict or understand? -> when is the next feed going to happen based on historical data
- What is one row in my dataset? -> Feed start timestamp | Feed end timestamp | Duration
- What is the target variable? -> minutes_until_next_feed
- Is this:
  - ~~Classification → predict a category~~
  - ~~Regression → predict a number~~
  - ~~Clustering → discover groups~~
  - **Time series → predict future values**
- What would make the model useful? -> Accurate predictions (even though this will never be completely accurate but a fun project) when the next feed is likely to happen. For ex. when leaving from home for an hour checking "Oh the next feed is likely to be within the next 30 minutes" so you might try to feed before leaving. Also possibly predicting duration but not the main focus now.