# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 1: Standardized Testing, Statistical Summaries and Inference

## Executive Summary

### Overview

The Ames housing data contains many recorded features of homes in Ames, IA. Our goal is to create a regression model that accuratley predicts the price of houses in Ames,IA. All the while keeping in mind certain important reccomendations for home buyers when investing in a home or selling their home. 

### Problem Statement

This technical report steps through the workflow of building a predictive model to more accuratley predict house prices in Ames, IA. Multiple rgression models (Linear rgression, lasso regression and KNN regression) are evaluated in order to select a production model with the least amount of error (e.g. bias) that generalizes well to the data (.e.g low variance b/w test and train scores). 

Throughout the report we'll seek to identify trends in the data and answers relevant questions to homeowenrs such as which features add the most value to a home, which features hurt home value the most, things homeowners can do to improve the value of their homes, preferred neighborhoods and whether or not this model would work across other cities. 


### Contents of this README

- Executive Summary
- Data-set Description
- Primary Findings and Insights
- Conclusions and Recommendations
- Next Steps

---

### Data-set Overview

#### Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|Id|int64|train,test|Unique record identifier|
|PID|int64|train,test|Alternative unique record identifier|
|MS SubClass|int64|train,test|The building class (e.g. Duplex, 1-Story 1946 & Newer All Styles etc.)|
|MS Zoning|object|train,test|Identifies the general zoning classification of the sale (e.g. A - Agirculture, C - Commercial, etc)|
|Lot Frontage|float64|train,test|Linear feet of street connected to property|
|Lot Area|int64|train,test|Lot size in square feet|
|Street|object|train,test|Type of road access to property (Gravel or Paved)|
|Alley|object|train,test|Type of alley access to property|
|Lot Shape|object|train,test|General shape of property|
|Land Contour|object|train,test|Flatness of the property|
|Utilities|object|train,test|Type of utilities available|
|Lot Config|object|train,test|Lot configuration|
|Land Slope|object|train,test|Flatness of the property|
|Neighborhood|object|train,test|Physical locations within Ames city limits|
|Condition 1|object|train,test|Proximity to main road or railroad|
|Condition 2|object|train,test|Proximity to main road or railroad (if second is present)|
|Bldg Type|object|train,test|Type of dwelling|
|House Style|object|train,test|Style of dwelling
|Overall Qual|int64|train,test|Overall material and finish quality|
|Overall Cond|int64|train,test|Overall condition rating|
|Year Built|int64|train,test|Original construction date|
|Year Remod/Add|int64|train,test|Remodel date (same as construction date if no remodeling or additions)|
|Roof Style|object|train,test|Type of roof|
|Roof Matl|object|train,test|Roof material|
|Exterior 1st|object|train,test|Exterior covering on house|
|Exterior 2nd|object|train,test|Exterior covering on house (if more than one material)|
|Mas Vnr Type|object|train,test|Masonry veneer type|
|Mas Vnr Area|float64|train,test|Masonry veneer area in square feet|
|Exter Qual|object|train,test|Exterior material quality|
|Exter Cond|object|train,test|Present condition of the material on the exterior|
|Foundation|object|train,test|Type of foundation|
|Bsmt Qual|object|train,test|Height of the basement|
|Bsmt Cond|object|train,test|General condition of the basement|
|Bsmt Exposure|object|train,test|Walkout or garden level basement walls|
|BsmtFin Type 1|object|train,test|Quality of basement finished area|
|BsmtFin SF 1|int64|train,test|Type 1 finished square feet|
|BsmtFin Type 2|object|train,test|Quality of second finished area (if present)|
|BsmtFin SF 2|int64|train,test|Type 2 finished square feet|
|Bsmt Unf SF|int64|train,test|Unfinished square feet of basement area|
|Total Bsmt SF|int64|train,test|Total square feet of basement area|
|Heating|object|train,test|Type of heating|
|Heating QC|object|train,test|Heating quality and condition|
|Central Air|object|train,test|Central air conditions (Y/N)|
|Electrical|object|train,test|Electrical system|
|1st Flr SF|int64|train,test|First floor square feet|
|2nd Flr SF|int64|train,test|Second floor square feet|
|Low Qual Fin SF|int64|train,test|Low quality finished square feet (all floors)|
|Gr Liv Area|int64|train,test|Above grade(ground) living area square feet|
|Bsmt Full Bath|int64|train,test|Basement full bathrooms|
|Bsmt Half Bath|int64|train,test|Basement half bathrooms|
|Full Bath|int64|train,test|Full bathrooms above grade|
|Half Bath|int64|train,test|Half baths above grade|
|Bedroom AbvGr|int64|train,test|Number of bedrooms above basement level|
|Kitchen AbvGr|int64|train,test|Number of kitchens|
|Kitchen Qual|object|train,test|Kitchen quality|
|TotRms AbvGrd|int64|train,test|Total rooms above grade (does not include bathrooms)|
|Functional|object|train,test|Home functionality rating|
|Fireplaces|int64|train,test|Number of fireplaces|
|Fireplace Qu|object|train,test|Fireplace quality|
|Garage Type|object|train,test|Garage Location|
|Garage Yr Blt|float64|train,test|Year garage was built|
|Garage Finish|object|train,test|Interior finish of the garage|
|Garage Cars|int64|train,test|Size of garage in car capacity|
|Garage Area|int64|train,test|Size of garage in square feet|
|Garage Qual|object|train,test|Garage quality|
|Garage Cond|object|train,test|Garage condition|
|Paved Drive|object|train,test|Paved driveway|
|Wood Deck SF|int64|train,test|Wood deck area in square feet|
|Open Porch SF|int64|train,test|Open porch area in square feet|
|Enclosed Porch|int64|train,test|Enclosed porch area in square feet|
|3Ssn Porch|int64|train,test|Three season porch area in square feet|
|Screen Porch|int64|train,test|Screen porch area in square feet|
|Pool Area|int64|train,test|Pool area in square feet|
|Pool QC|object|train,test|Pool quality|
|Fence|object|train,test|Fence quality|
|Misc Feature|object|train,test|Miscellaneous feature not coveed in other categories (Elevator, Shed, etc)|
|Misc Val|int64|train,test|$ Value of miscellaneous feature|
|Mo Sold|int64|train,test|Month sold|
|Yr Sold|int64|train,test|Year sold|
|Sale Type|object|train,test|Type of sale|
|SalePrice|int64|test|Price of sale|

*A more detailed data dictionary exists on the Kaggle website [here](https://www.kaggle.com/c/dsi-us-11-project-2-regression-challenge/data)

#### Provided Data

For this project, I've used the following two datasets as source data:

- [Test Data](./datasets/test.csv)
- [Training Data](./datasets/train.csv)
- [Sample Output](./datasets/sample_sub_reg.csv)

You can see the sources for the test and train data [here](https://www.kaggle.com/c/dsi-us-11-project-2-regression-challenge/data)

Additionally, I've created the following output dataset for submission to the Kaggle website.

- [Submission Data](./datasets/submission_lasso.csv)


#### Primary Findings & Insights

I selected the lasso regression model as my production model given it's ability select the most impactful data features on the target vector of sales price and tune-out the noise of the remaining features. This is an import feature of the lasso model and I chose to leverage it here given my lower scoring intial results with feature engineering and linear regression models as well as the shear amount of features in this dataset (>80). 

The production model I selected was a ridge model that scored ~88.4% & 88.5% respectively on the train-test split in the training data and shows an 85% cross-validation score on the entire training dataset. When compared to the lasso model I originally submited, this model indicated it would generalize better and have lower variance (~90% train and 87% test). The first submission lasso model predicted home prices with a 35970.80162 RMSE whereas the second submission ridge model predicted home prices with a 37449.86822 RMSE in the Kaggle competition. Althought the RMSE was higher in the final compatition, I do believe the ridge model will generalize better to data in other cities and have selected it as my production model.

Through exploratory analysis of the model coefficients it is clear that the features with the largest impact on home prices are unsurprisingly: 

Related to the total size of the home (houses are traditionally valued by the sqft)
- Lot area
- Total basement sqft
- 1st floor sqft
- 2nd floor sqft
- Ground living area 
- Garage # of cars

Related to the age, quality and conditon of the home
- Overall quality of the home
- Overall condition of the home
- Year built
- Year re-modelled
- Home functionality rating

Related to additional upgrades or desirable features in the home
- Basement full bath
- Number of fireplaces
- Miscaleanous features valus (Tennis court, 2nd Garage, large shed, etc)
- Roof Material
- Number of Full Baths
- Total rooms above ground
- Screen Porch

*Caveat to this is that the model shows these features are correlated to higher prices but that does not necissarily indicated their relationships is causal. It could be possible that large highly priced homes with higher sqft just happen to already have these features on average. The model could be picking up and predicting this trend.

Although the model does not assign a large coefficient to neighborhood, this could very well be due to the fact that the neighborhoods were discrete cateogrical and not continuous variables. Given what we all know to be true of real estate in the real world, if we had median saleprice information by neighbordhood or a neighborhood score, I believe this category would have had a much higher weighting in the model. As we all know from our real estate agents: Location! Location! Locaton! Given that it is important to callout the most desirable neighborhoods in the Ames city limits for prospective investors. The top 5 neighbordhoods by mean salesprice are:

1. Stone Brooke - Avg saleprice of $329676
2. Northridge Heights - Avg saleprice of $322831
3. Northridge - Avg saleprice of $316294
4. Green Hill - Avg saleprifce of $280000
5. Veenker - Avg saleprice of $253571

*Caveate here is that this information would need to be removed in order for the model to generalize well to other cities. 

#### Conclusion & Recommendations

For any homebuyers or sellers looking to transact in the Ames city limits, it is important to consider the total size of your home (sqft), the age/quality/condition of the home as well as any additional upgrades or desirable features that add value to the home. Assuming it works within their budget, I would reccomend for them to first look for homes in the Stone Brook, Northridge Heights, Northridge, Green Hill and Veenker neighborhoods prior to searching for homes in other areas. 

#### Next Steps

Moving onto the next phase of this analysis would require additional time. In order to further refine the data, it would ne necissary to conduct additional feature engineering (possibly using polynomials) and then utilize both the PIPE methods and Gridsearch methods to further refine the most important features and hyperparameters in the dataset. Given more time I would ensure to utilize the PIPE method to exhaust all feature types and then leverage the Gridsearch method to fine tune multiple algorithms in order to identify the highest performing one given these updated features. 