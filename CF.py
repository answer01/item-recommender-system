import pandas as pd
import math
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from random import randint

def generateRandomItems(Reviews, NUM_OF_ITEMS, COUNT_LIMIT):
    """
        Sorts items by number of users that rated the product.
        Depending on NUM_OF_ITEMS and lower limit COUNT_LIMIT
        returns random itemIDs
    """
    # group by ProductId and count all users that rated that exact product
    numOfRated = Reviews.groupby(['ProductId']).agg({'UserId':['count']})
    numOfRated.columns = ['Count']
    # sorting values 
    numOfRated = numOfRated.sort_values(by = 'Count', ascending = False)

    # leave values in dataframe if they meet the COUNT_LIMIT priority
    numOfRated = numOfRated[numOfRated['Count'] > COUNT_LIMIT]
    numberOfItems = numOfRated.shape[0]
    print(numberOfItems)

    for i in range(0, NUM_OF_ITEMS):
        itemIndex = randint(0, numberOfItems-1)
        itemID = numOfRated.iloc[[itemIndex]].index.tolist()[0]
        ItemIDs.append(itemID)
    return ItemIDs    
    
def fillDictionary(ReviewsD):
    """
        Fills the input from .csv file into two key dict
        {key:{key1:value, key2,value, ....}}
    """
    for row in Reviews.itertuples():
        itemID = row[1]
        # if itemID already exists in ReviewsD dictionary
        if (itemID in ReviewsD):
            userScoreD1 = ReviewsD[itemID]
            userScoreD1[row[2]] = row[3]
            ReviewsD[itemID] = userScoreD1
            continue
        # creating new dictionary if theres a new itemID
        userScoreD = {}
        userScoreD[row[2]] = row[3]
        ReviewsD[itemID] = userScoreD

def averageScore(userScoreD):
    """
        userScoreD is the dictionary that contains user scores.
        Returns the average user's score on a single item
    """
    sum = 0
    len = 0
    # go over scores in {userID:score,...} dictionary
    for score in userScoreD.values():
        sum += score
        len += 1
    # return average users score for an item
    return sum/len
 
def calculateSimmilarities(ReviewsD, itemIDTest, userScore):
    """
        ReviewsD is the dictionary that contains entire csv except
        the item that simmilarity is calculated on.
        itemIDTest and userScore are key and value for that item.
        Function finds the items that were rated by the same user
        and sums it for Pearsons correlation calculation
    """
    # go over entire dictionary without testing(removed) item
    for itemID, userScoreOther in ReviewsD.items():
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sim = 0
        # go over user:score pairs that rated test item
        for testUser, testScore in userScore.items():
            # if items were rated by same user
            if testUser in userScoreOther:
                # get the value from the other item
                otherScore = userScoreOther[testUser]
                sim = 1
                # calculate all the sums
                avgScoreTest = averageScore(userScore)
                avgScoreOther = averageScore(userScoreOther)
                sum1 += ((testScore-avgScoreTest)*(otherScore-avgScoreOther))
                sum2 += math.pow((testScore-avgScoreTest), 2)
                sum3 += math.pow((otherScore-avgScoreOther), 2)
        # if simmilar items were found, calculate the simmilarity
        if (sim == 1):
            sum2 = math.sqrt(sum2)
            sum3 = math.sqrt(sum3)
            # if error
            if (np.isnan((sum1)/(sum2*sum3))):
                continue
            # add to dictionary, using productID as key
            simmilarities[itemID] = (sum1)/(sum2*sum3)
    # return {itemID:sim,....} dictionary
    return simmilarities       

def calculatePredictions(ReviewsD, userIDTest, scoreTest, simmilarities):
    """
        Function finds userIDTest in all simmilar items and uses all the
        scores for prediction calculation
        Returns actualScore and predictedScore for further calculations
        of finding rmse and mse values
    """
    score = 0
    sim = 0    
    sumB = 0
    sumN = 0
    # go over entire dictionary without testing(removed) item
    for itemID, userScoreOther in ReviewsD.items():
        # if same users were found
        if (userIDTest in userScoreOther):
            # find simmilarity and score
            if (itemID in simmilarities):
                sim = simmilarities[itemID]
                if (sim == -1):
                    continue
                score = userScoreOther[userIDTest]
            # calculations for prediction
            sumB += (score*sim)
            sumN += math.fabs(sim)
    if (sumB != 0 and sumN != 0):
        print("User: ", userIDTest)
        print("Actual score: ", scoreTest)
        print("Predicted score: ", math.fabs(sumB/sumN))
        actualScore = scoreTest
        predictedScore = math.fabs(sumB/sumN)
        print(" ")
        # if predictions are found
        return (actualScore, predictedScore)   
    else:
        # no predictions found
        return None
######################################################


if __name__ == "__main__":
    
    Reviews = pd.read_csv('Reviews.csv')

    ReviewsD = {}
    simmilarities = {}
    ItemIDs = []
    rmse = 0
    mae = 0
    NUM_OF_ITEMS = 10
    COUNT_LIMIT = 10

    # remove unnecessary columns
    del Reviews['HelpfulnessNumerator']
    del Reviews['HelpfulnessDenominator']
    del Reviews['Time']
    del Reviews['Text']
    del Reviews['Id']
    del Reviews['ProfileName']
    del Reviews['Summary']

    ItemIDs = generateRandomItems(Reviews, NUM_OF_ITEMS, COUNT_LIMIT)
    fillDictionary(ReviewsD)

    for itemID in ItemIDs:
        print('----------------------------------------')
        print('Starting predictions for Item: ', itemID)
        print('----------------------------------------')
        actualScores = []
        predictedScores = []
        userScore = ReviewsD.pop(itemID)
        simmilarities = calculateSimmilarities(ReviewsD, itemID, userScore)
        for userID, score in userScore.items():
            if (calculatePredictions(ReviewsD, userID, score, simmilarities) is None):
                continue
            else:
                (actualScore, predictedScore) = calculatePredictions(ReviewsD, userID, score, simmilarities)
                actualScores.append(actualScore)
                predictedScores.append(predictedScore)
        ReviewsD[itemID] = userScore
        if (not actualScores or not predictedScores):
            continue
        rmse += math.sqrt(mean_squared_error(actualScores, predictedScores))
        mae += mean_absolute_error(actualScores, predictedScores)

    print('----------------------------------------')
    print(rmse, mae)
    print("Root mean squared error: ", rmse/NUM_OF_ITEMS)
    print("Mean absolute error: ", mae/NUM_OF_ITEMS)
    print('----------------------------------------')