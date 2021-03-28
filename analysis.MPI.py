import json
from mpi4py import MPI
import math
import pandas as pd
import re
import ijson
from functools import reduce
import argparse
pd.set_option('display.max_columns', None)  ## display all columns of out output

#########################################
# param:
#     address: the address of affin matching vocabs
# return:
#     afinn: the dict to map vocabs into sentimental score
########################################## 
def generate_Affin_Dict(address):
    afinn = {}
    with open(address, 'r') as f:
        for line in f.readlines(): # Read by lines
            kv = line.strip().split()
            # for the splitted words, except for the last word as score, the rest words make up the token to be matched
            token = ' '.join(kv[0:len(kv) - 1]) 
            num = 0
            if kv[-1][0] == '-':
                num = -1 * int(kv[-1][1:]) # handle the conversion of negative 
            else:
                num = int(kv[-1])
            afinn[token] = num # append into the dict
    return afinn

##########################################################
# param:
#     address: the address of melbourne grid data
# return:
#     grid: the nested dict of grid information. A grid is conceptualized 
#           by its xmin, xmax, ymin, ymax properties and named by Id
########################################################### 
def generate_grid_dict(address):
    grid = {}
    with open(address, 'r') as f:
        data = json.load(f)
        for feature in data['features']:
            # properties = [id, xmin, xmax, ymin, ymax]
            ID = feature['properties']['id']
            coords = {}
            for key, value in feature['properties'].items():
                if key != 'id':
                    coords[key] = value
            grid[ID] = coords
    return grid


###########################################################################
# param:
#     data: the data to be processed on each progress
#     afinn: the dict to map sentimental vocabs into num
#     grid: the dict of grid information 
# return:
#     sentiment_sums: the sum-up sentimental scores for each grid. 
#                     Eg. Cell #Total Tweets, #Overall Sentiment Score
#                           A1          1234                        25
############################################################################
def calculate_senti_sum_in_parallel(data, afinn, grid):
    sentiment_sums = []
    for i in range(len(grid.keys())):
        sentiment_sums.append([mapping_id_to_gc(i), 0, 0]) # create 15 rows of grids
    for tweet in data:
        location = tweet['location']
        grid_code = find_grid(location, grid) # Get grid code according to location
        if grid_code != '': # Skip if no grid code found
            sentiment_sums[mapping_gc_to_id(grid_code)][1] += 1 # tweet count adding up
            text = tweet['text'].lower() # handle sensitive cases by lowercase
            match_sentimental_words(text, afinn, sentiment_sums, grid_code) # match vocabs with afinn dict and add up sentimental scores
    sentiment_sums = pd.DataFrame(sentiment_sums, columns=['Cell', '#Total Tweets', '#Overall Sentiment Score']) # output the dataframe result
    return sentiment_sums

################################################################# 
# param:
#     location: the location data of a tweet
#     grid: the dict of grid information 
# return:
#     grid code: the code name of grid where the denoted location is at
##################################################################  
def find_grid(location, grid):
    xPos = location[0]
    yPos = location[1]
    grid_code = ''
    for key, value in grid.items():
        xMin = value['xmin']
        xMax = value['xmax']
        yMin = value['ymin']
        yMax = value['ymax']
        if xPos >= xMin and xPos < xMax and yPos > yMin and yPos <= yMax:
            # Requirement: B1/B2 choose B1, the left;
            #              B2/C2 choose C2, the below;
            # B1(xmin = 144.7, xmax = 144.85), B2(xmin = 144.85, xmax = 145)
            # So if x = 144.85 which is x >= xmin, assign grid code B2
            # B2(ymin = -37.8, ymax = -37.65), C2(ymin = -37.95, xmax = -37.8)
            # So if y = -37.8 which is y <= ymax, assign grid code C2
            return key
    return grid_code


#################################################################
# param:
#     text: the sentence of the tweet
#     afinn: the dict to map sentimental vocabs into num
#     sentiment_sums: the sum of sentimental score in tweets
#     grid_code: the code name of grid
# return:
#     tokenize the sentence as required and adding up the sentimental score
#     of words which are matched by affin dict
###################################################################
def match_sentimental_words(text, afinn, sentiment_sums, grid_code):
    words = re.split(r'[\s\,\.\!\?\'\"]+', text) # special endings ,.!?'"
    i = 0
    while i < len(words): # MaxMatch algorithm
        selected_words = words[i:] # intercepted words list from the original
        temp = selected_words[0] # start from the first word
        matched_word = '' if not temp in afinn.keys() else temp # word in match
        j = 1
        while j < len(selected_words):
            temp = ' '.join([temp, selected_words[j]]) # 'can't' + 'stand' = 'can't stand'
            j += 1
            if temp in afinn.keys():
                matched_word = temp # update word in match
        if matched_word == '':
            i += 1 # no match found and shift index to right
        else:
            sentiment_sums[mapping_gc_to_id(grid_code)][2] += afinn[matched_word]
            i += j # match found and shift index to right by length j


#########################################
# param:
#     grid_code: the code name of grid
# return:
#     the id representing the grid code    
##########################################
def mapping_gc_to_id(grid_code):
    gc_map = {'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 
              'B1': 4, 'B2': 5, 'B3': 6, 'B4': 7,
              'C1': 8, 'C2': 9, 'C3': 10, 'C4': 11, 'C5': 12,
              'D3': 13, 'D4': 14, 'D5': 15}
    return gc_map[grid_code]

#########################################
# param:
#     gc_id: the id representing the grid code
# return:
#     gthe code name of grid   
##########################################
def mapping_id_to_gc(gc_id):
    id_map = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4',
              'C1', 'C2', 'C3', 'C4', 'C5', 'D3', 'D4', 'D5']
    return id_map[gc_id]

##################################################################
# param:
#     path: the path of data to read
#     size: MPI.COMM_WORLD.Get_size()
#     comm: the MPI communicator
#     rank: the rank of this MPI task
#     afinn: the dict to map sentimental vocabs into num
#     grid: the dict of grid information 
# return:
#     Iterately read json file, almost evenly split the twitter data,
#     assign data to each active thread and collect the computed results      
################################################################### 

def split_data_and_process(path, size, comm, rank, afinn, grid):
    data_to_share = [] # list for storing the shared tweet data
    tweet = {} # empty tweet 
    senti_sums_splits = [] # the splitted scores
    with open(path, 'r', encoding='utf-8') as f:
        for prefix, event, value in ijson.parse(f): # read json iterately
            if prefix.endswith('.properties.text'):
                tweet['text'] = value # append the text of tweet
            if prefix.endswith('.geometry.coordinates.item'):
                if not 'location' in tweet.keys():
                    tweet['location'] = [value] # append the xpos of tweet
                else:
                    tweet['location'].append(value) # append the ypos of tweet
            if 'location' in tweet.keys() and 'text' in tweet.keys() and len(tweet['location']) == 2:
                # apend the tweet into share list only if it is valid
                data_to_share.append(tweet) 
                tweet = {}
            if len(data_to_share) == size * 100:
                datas = [data_to_share[i * 100 : (i + 1) * 100] for i in range(size)] # create the iterable data list for sharing
                data_to_process = comm.scatter(datas, root=0) # scatter data from rank 0 task to all active tasks
                senti_sum = calculate_senti_sum_in_parallel(data_to_process, afinn, grid) # paraller commputing sentimental scores
                senti_sums_split = comm.reduce(senti_sum, op=gather_result, root=0) # collected the reduced computation results
                if rank == 0:
                    senti_sums_splits.append(senti_sums_split) # append the the compuation result of this batch
                data_to_share = [] # clear the share list and restart
    if len(data_to_share) > 0: # Do the last parallel computing if data is left 
        datas = [data_to_share[i * 100 : (i + 1) * 100] for i in range(size)]
        data_to_process = comm.scatter(datas, root=0)
        senti_sum = calculate_senti_sum_in_parallel(data_to_process, afinn, grid)
        senti_sums_split = comm.reduce(senti_sum, op=gather_result, root=0)
        if rank == 0:
            senti_sums_splits.append(senti_sums_split)
    if rank == 0:
        senti_sums_total = reduce(gather_result, senti_sums_splits) # reduce adding up the splitted tweet counts and tweet scores
        senti_sums_total['#Overall Sentiment Score'] = senti_sums_total['#Overall Sentiment Score'].apply(intToPositiveStr) # change int to string with signs
        print('The sentimental sum of twitter dataset is:')
        print(senti_sums_total)



###################################################
# param:
#     x: the first sentiment sums 
#     y: the second sentiment sums
# return:
#     x: the new sentiment sums which merges the scores
#        between previous x and y     
#################################################### 
def gather_result(x,y):
    x['#Total Tweets'] = x['#Total Tweets'] + y['#Total Tweets']
    x['#Overall Sentiment Score'] = x['#Overall Sentiment Score'] + y['#Overall Sentiment Score']
    return x

###################################################
# param:
#     num: a number, maybe positive or negative 
# return:
#     the signed string of number
#     e.g 20 -> '+20', 0 -> '0', -15 -> '-15'     
####################################################
def intToPositiveStr(num):
    return '+' + str(num) if num > 0 else str(num)


# create python script commands for entering json data address
parser = argparse.ArgumentParser(description='Path of files to be processed')
parser.add_argument('--grid', type=str,  default='melbGrid.json', help='Path to Melbourne Grid json file')
parser.add_argument('--afinn', type=str,  default='AFINN.txt', help='Path to AFINN text file')
parser.add_argument('--tweet', type=str,  default='tinyTwitter.json', help='Path to Twitter data json')
args = parser.parse_args()

def main():
    comm = MPI.COMM_WORLD  # the global communicator
    rank = comm.Get_rank() # the index of member
    size = comm.Get_size() # the total number of members

    afinn, grid, data_to_share = None, None, None
    
    if rank == 0 :
        afinn = generate_Affin_Dict(args.afinn)
        grid = generate_grid_dict(args.grid)

    afinn = comm.bcast(afinn, root=0) # broadcast afinn dict to the other members of the group
    grid = comm.bcast(grid, root=0) # broadcast grid dict to the other members of the group
    
    split_data_and_process(args.tweet, size, comm, rank, afinn, grid)

if __name__ == "__main__":
    main()
