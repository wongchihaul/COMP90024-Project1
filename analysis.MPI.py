import json
from mpi4py import MPI
import math
import pandas as pd
import re
from functools import reduce
import argparse
from datetime import datetime
import mmap
pd.set_option('display.max_columns', None)  ## display all columns of out output

#########################################
# param:
#     address: the address of affin matching vocabs
#     comm: the MPI communicator
#     size: MPI.COMM_WORLD.Get_size()
#     rank: the rank of this MPI tas
# return:
#     afinn: the dict to map vocabs into sentimental score
########################################## 
def generate_Affin_Dict(address, comm, size, rank):
    afinn = {}
    with open(address, 'r') as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            for line in iter(mm.readline, b''): # Read by lines
                line = str(line, encoding='utf-8')
                kv = re.split(r'[\s"]+', line.strip())
                # for the splitted words, except for the last word as score, the rest words make up the token to be matched
                token = ' '.join(kv[0:len(kv) - 1]) 
                num = int(kv[-1])
                afinn[token] = num # append into the dict
    return afinn


##########################################################
# param:
#     address: the address of melbourne grid data
#     comm: the MPI communicator
#     size: MPI.COMM_WORLD.Get_size()
#     rank: the rank of this MPI tas
# return:
#     grid: the nested dict of grid information. A grid is conceptualized 
#           by its xmin, xmax, ymin, ymax properties and named by Id
########################################################### 
def generate_grid_dict(address, comm, size, rank):
    grid = {}
    with open(address, 'r') as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            for line in iter(mm.readline, b''):
                line = str(line, encoding='utf-8') 
                if line.endswith('] ] ] } },\n'):
                    line = json.loads(line[:-2])
                elif line.endswith('] ] ] } }\n'):
                    line = json.loads(line[:-1]) 
                else:
                    continue    
                feature = line 
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
#     sentiment_sums: the sum-up sentimental scores for each grid. 
#                     Eg. Cell #Total Tweets, #Overall Sentiment Score
#                           A1          1234                        25
#     tweet: the tweet data to be processed on each progress
#     afinn: the dict to map sentimental vocabs into num
#     grid: the dict of grid information 
# return:
#     the new sentiment_sums after matching words and adding scores
############################################################################
def calculate_senti_sum_in_parallel(senti_sums, tweet, afinn, grid):
    # for tweet in tweets:
    location = tweet['value']['geometry']['coordinates']
    grid_code = find_grid(location, grid) # Get grid code according to location
    if grid_code != '': # Skip if no grid code found
        senti_sums[mapping_gc_to_id(grid_code)][1] += 1 # tweet count adding up
        text = tweet['value']['properties']['text'].lower() # handle sensitive cases by lowercase
        match_sentimental_words(text, afinn, senti_sums, grid_code) # match vocabs with afinn dict and add up sentimental scores

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
        if xPos > xMin and xPos <= xMax and yPos > yMin and yPos <= yMax:
            # Requirement: B1/B2 choose B1, the left;
            #              B2/C2 choose C2, the below;
            # B1(xmin = 144.7, xmax = 144.85), B2(xmin = 144.85, xmax = 145)
            # So if x = 144.85 which is x <= xmax, assign grid code B1
            # B2(ymin = -37.8, ymax = -37.65), C2(ymin = -37.95, ymax = -37.8)
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
    words = re.split(r'[\s]+', text) # split sentence into words using re
    special_endings = [',', '.', '!', '?', '\'', '\"'] # special endings ,.!?'"
    i = 0
    while i < len(words):  # MaxMatch algorithm
        selected_words = words[i:]
        matched_word = None
        j = 1
        wl = 1
        maxW = 2
        while j <= len(selected_words) and j <= maxW:
            temp = ' '.join(selected_words[:j])
            if len(temp) >= 1 and temp[-1] in special_endings:
                temp = re.match(r'^(.*?)([\,\.\?\!\'\"]+)$', temp).group(1)
            # while len(temp) >= 1 and temp[-1] in special_endings:
            #     temp = temp[:-1]
            if temp in afinn.keys():
                matched_word = temp
                wl = j
            j += 1
        if not matched_word:
            i += 1
        else:
            sentiment_sums[mapping_gc_to_id(grid_code)][2] += afinn[matched_word]
            i += wl


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
#     the code name of grid   
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
    # data_to_process = [] # list for storing the shared tweet data
    # senti_sums_splits = [] # the splitted scores
    # m = 1000
    senti_sums_chunk = []
    for i in range(len(grid.keys())):
        senti_sums_chunk.append([mapping_id_to_gc(i), 0, 0]) # create 15 rows of grids
    with open(path, 'r', encoding='utf-8') as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            offset = mm.size() / size # almost evenly split data
            end_line = None # find stop point
            if rank < size - 1:
                mm.seek(int((rank + 1) * offset))
                mm.readline()
                end_line = str(mm.readline(), encoding='utf-8')
            mm.seek(int(rank * offset)) # jump to start point
            #print('Rank %s start at %s of %s' % (rank, mm.tell(), mm.size()))
            i = 0
            for line in iter(mm.readline, b''): # iteratively reading
                line = str(line, encoding='utf-8')
                if rank != 0 and i == 0:
                    i += 1
                    continue
                if rank < size - 1 and line == end_line:
                    break                 
                if line.endswith('}},\r\n'):
                    line = json.loads(line[:-3])
                elif line.endswith('}}]}\r\n'):
                    line = json.loads(line[:-4]) 
                elif line.endswith('}}\r\n'):
                    line = json.loads(line[:-2])
                else:
                    continue    
                tweet = line 
                calculate_senti_sum_in_parallel(senti_sums_chunk, tweet, afinn, grid)
    senti_sums_chunk = pd.DataFrame(senti_sums_chunk, columns=['Cell', '#Total Tweets', '#Overall Sentiment Score'])
    senti_sums_total = comm.reduce(senti_sums_chunk, op=gather_result, root=0)
    if rank == 0:
        # merge results to rank 0 thread and print out
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
    
    afinn = generate_Affin_Dict(args.afinn, comm, size, rank)
    grid = generate_grid_dict(args.grid, comm, size, rank)

    split_data_and_process(args.tweet, size, comm, rank, afinn, grid)

if __name__ == "__main__":
    main()
