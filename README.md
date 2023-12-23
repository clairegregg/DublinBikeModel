# DublinBikes Machine Learning

This is the final project for my Machine Learning module where I will study the impact of COVID-19 on the usage of Dublin Bikes.

## Steps in the project

1. Data preprocessing & display
2. Feature Engineering
    - We need to know *bike usage* not bike/bike stand availability
3. Assess impact of pandemic on city-bike usage for pandemic period
    - Predict how city-bike usage might have been if the pandemic had not happened
4. Assess impact of pandemic on city-bike usage for post-pandemic period:
    - Define when post-pandemic period is!
    - Predict how city-bike usage might have been if the pandemic had not happened

## Assignment Brief

### General Instructions
Download the Dublinbikes dataset at [https://data.gov.ie/dataset/dublinbikesapi](https://data.gov.ie/dataset/dublinbikesapi). The dataset is organised in part quarterly (i.e., four data-files per year) and in part monthly, meaning that the project might involve concatenating different portions of the dataset. Note that there might be many possible answers to each of the questions that follow, so you will need to be creative and think about this on your own.

### Scenario
You are working for FUTURE-DATA a local company specialised in data science. Dublin City Council hired your company to study the impact of COVID-19 on the city-bikes usage as they are planning to optimise the city-bike system. Dublin City Council had originally structured the city-bike network based on the forecasts of bike usage up to 2030. However, they think that the usage may not match the initial prediction because of the impact of the pandemic on our mobility. FUTURE-DATA decided that their first step should be to investigate the impact of the pandemic on the usage of the city bike network.

### Task
The company agreed with our manager on two goals:
1. To assess the impact of the pandemic on the city-bike usage for the pandemic period.
2. To assess the impact of the pandemic on the city-bike usage for the post-pandemic period.

**Hint 1:** Note that there are many ways to do this and angles to explore. Temporal vs. spatial dynamics
might have changed. A first approach might be to use descriptive statistics only. But it is also required
to use machine learning to estimate how the city-bike usage would have been if the pandemic had
not happened. Predictions can be augmented by including information that is not available in the
Dublin-bikes dataset (e.g., weather data).

**Hint 2:** The original features tell us about bike and bike stand availability. However, that is a different
concept from “bike usage” i.e., how many bikes have been taken from (or brought to) that station.
We suggest deriving a “bike usage” features for the analyses. Other ways of tackling the tasks are also
accepted, but remember to justify your choices and discuss your results clearly. Please also remember
to report clear, compact figures.

*indicative breakdown of mark: (i) data preprocessing and feature engineering 20 marks, (ii) machine learning methodology 20 marks, (iii) evaluation 25 marks, (iv) report presentation 10 marks*