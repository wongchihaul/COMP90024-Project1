import json
from mpi4py import MPI
import math
import pandas as pd
import re
import ijson
from functools import reduce
import argparse
from datetime import datetime
import mmap
pd.set_option('display.max_columns', None)  ## display all columns of out output

#########################################
# param:
#     address: the address of affin matching vocabs
# return:
#     afinn: the dict to map vocabs into sentimental score
#     maxW: the max number of words combination
########################################## 
def generate_Affin_Dict(address, comm, size, rank):
    afinn = {}
    with open(address, 'r') as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            offset = mm.size() / size
            mm.seek(int(rank * offset))
            i = 0
            start_lines = None
            for line in iter(mm.readline, b''): # Read by lines
                line = str(line, encoding='utf-8')
                if rank != 0 and i == 0:
                    i += 1
                    continue
                if not start_lines:
                    start_lines = {rank : line}
                    start_lines = comm.allreduce(start_lines, op=gather_dict)
                if rank < size - 1 and line == start_lines[rank + 1]:
                    break
                kv = re.split(r'[\s"]+', line.strip())
                # for the splitted words, except for the last word as score, the rest words make up the token to be matched
                token = ' '.join(kv[0:len(kv) - 1]) 
                num = int(kv[-1])
                afinn[token] = num # append into the dict
    afinn_total = comm.allreduce(afinn, op=gather_dict)
    return afinn_total

##########################################################
# param:
#     address: the address of melbourne grid data
# return:
#     grid: the nested dict of grid information. A grid is conceptualized 
#           by its xmin, xmax, ymin, ymax properties and named by Id
########################################################### 
def generate_grid_dict(address, comm, size, rank):
    grid = {}
    with open(address, 'r') as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            offset = mm.size() / size
            mm.seek(int(rank * offset))
            i = 0
            start_lines = None
            for line in iter(mm.readline, b''):
                line = str(line, encoding='utf-8')
                if rank != 0 and i == 0:
                    i += 1
                    continue
                if not start_lines:
                    start_lines = {rank : line}
                    start_lines = comm.allreduce(start_lines, op=gather_dict)
                if rank < size - 1 and line == start_lines[rank + 1]:
                    break 
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
    grid_total = comm.allreduce(grid, op=gather_dict)
    return grid_total


###########################################################################
# param:
#     data: the data to be processed on each progress
#     afinn: the dict to map sentimental vocabs into num
#     grid: the dict of grid information 
#     maxW: the max number of words combination
# return:
#     sentiment_sums: the sum-up sentimental scores for each grid. 
#                     Eg. Cell #Total Tweets, #Overall Sentiment Score
#                           A1          1234                        25
############################################################################
def calculate_senti_sum_in_parallel(senti_sums, tweet, afinn, grid):
    # for tweet in tweets:
    location = tweet['value']['geometry']['coordinates']
    grid_code = find_grid(location, grid) # Get grid code according to location
    if grid_code != '': # Skip if no grid code found
        senti_sums[mapping_gc_to_id(grid_code)][1] += 1 # tweet count adding up
        text = tweet['value']['properties']['text'].lower() # handle sensitive cases by lowercase
        match_sentimental_words(text, afinn, senti_sums, grid_code) # match vocabs with afinn dict and add up sentimental scores
    # sentiment_sums = pd.DataFrame(sentiment_sums, columns=['Cell', '#Total Tweets', '#Overall Sentiment Score']) # output the dataframe result
    # return senti_sums

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
#     maxW: the max number of words combination
# return:
#     tokenize the sentence as required and adding up the sentimental score
#     of words which are matched by affin dict
###################################################################
def match_sentimental_words(text, afinn, sentiment_sums, grid_code):
    words = re.split(r'[\s]+', text) # split sentence into words using re
    special_endings = [',', '.', '!', '?', '\'', '\"'] # special endings ,.!?'"
    i = 0
    # while i < len(words):  # MaxMatch algorithm
    #     selected_words = words[i:]
    #     temp = selected_words[0]
    #     matched_word = '' if not temp in afinn.keys() else temp
    #     j = 1
    #     while j < len(selected_words) and j + 1 <= maxW:
    #         temp = ' '.join([temp, selected_words[j]])
    #         j += 1
    #         if temp in afinn.keys():
    #             matched_word = temp
    #     if matched_word == '':
    #         i += 1
    #     else:
    #         sentiment_sums[mapping_gc_to_id(grid_code)][2] += afinn[matched_word]
    #         i += len(matched_word.split())
    while i < len(words):
        matched_word = None
        for j in range(len(words), i, -1):  # Moving right pointer from tail to head
            temp = ' '.join(words[i:j])
            # if len(temp) >= 1 and temp[-1] in special_endings:
            #     temp = re.match(r'^(.*?)([\,\.\?\!\'\"]+)$', temp).group(1)
            while len(temp) >= 1 and temp[-1] in special_endings:
                temp = temp[:-1]
            if temp in afinn.keys():    # if matches, it can guarantee that this word is the maximum matching word 
                matched_word = temp
                break
        if matched_word is None:
            i += 1
        else:
            sentiment_sums[mapping_gc_to_id(grid_code)][2] += afinn[matched_word]
            i = j   # derived from i += j - i


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
#     maxW: the max number of words combination
# return:
#     Iterately read json file, almost evenly split the twitter data,
#     assign data to each active thread and collect the computed results      
################################################################### 

def split_data_and_process(path, size, comm, rank, afinn, grid):
    data_to_process = [] # list for storing the shared tweet data
    senti_sums_splits = [] # the splitted scores
    # m = 1000
    senti_sums_chunk = []
    for i in range(len(grid.keys())):
        senti_sums_chunk.append([mapping_id_to_gc(i), 0, 0]) # create 15 rows of grids
    with open(path, 'r', encoding='utf-8') as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            offset = mm.size() / size
            mm.seek(int(rank * offset))
            i = 0
            start_lines = None
            for line in iter(mm.readline, b''):
                line = str(line, encoding='utf-8')
                if rank != 0 and i == 0:
                    i += 1
                    continue
                if not start_lines:
                    start_lines = {rank : line}
                    start_lines = comm.allreduce(start_lines, op=gather_dict)
                if rank < size - 1 and line == start_lines[rank + 1]:
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
    #             data_to_process.append(tweet)
    #             if len(data_to_process) == size:
    #                 senti_sum = calculate_senti_sum_in_parallel(data_to_process, afinn, grid)
    #                 senti_sums_splits.append(senti_sum)
    #                 data_to_process = []
    # if len(data_to_process) > 0:
    #     senti_sum = calculate_senti_sum_in_parallel(data_to_process, afinn, grid)
    #     senti_sums_splits.append(senti_sum)
    #     data_to_process = []
    # senti_sums_chunk = reduce(gather_result, senti_sums_splits)
    senti_sums_chunk = pd.DataFrame(senti_sums_chunk, columns=['Cell', '#Total Tweets', '#Overall Sentiment Score'])
    senti_sums_total = comm.reduce(senti_sums_chunk, op=gather_result, root=0)
    if rank == 0:
        senti_sums_total['#Overall Sentiment Score'] = senti_sums_total['#Overall Sentiment Score'].apply(intToPositiveStr) # change int to string with signs
        print('The sentimental sum of twitter dataset is:')
        print(senti_sums_total)    

        # for tweet in ijson.items(f, 'rows.item'):
        #     data_to_share.append(tweet) 
        #     if len(data_to_share) == size * m:
        #         datas = [data_to_share[i * m : (i + 1) * m] for i in range(size)] # create the iterable data list for sharing
        #         data_to_process = comm.scatter(datas, root=0) # scatter data from rank 0 task to all active tasks
        #         senti_sum = calculate_senti_sum_in_parallel(data_to_process, afinn, grid) # paraller commputing sentimental scores
        #         senti_sums_split = comm.reduce(senti_sum, op=gather_result, root=0) # collected the reduced computation results
        #         if rank == 0:
        #             senti_sums_splits.append(senti_sums_split) # append the the compuation result of this batch
        #         data_to_share = [] # clear the share list and restart
    # if len(data_to_share) > 0: # Do the last parallel computing if data is left 
    #     chunk_size = int(math.ceil(float(len(data_to_share)) / size))
    #     datas = [data_to_share[i * chunk_size : (i + 1) * chunk_size] for i in range(size)]
    #     data_to_process = comm.scatter(datas, root=0)
    #     senti_sum = calculate_senti_sum_in_parallel(data_to_process, afinn, grid, maxW)
    #     senti_sums_split = comm.reduce(senti_sum, op=gather_result, root=0)
    #     if rank == 0:
    #         senti_sums_splits.append(senti_sums_split) 
    # if rank == 0:
    #     if len(data_to_share) > 0:
    #         senti_sum = calculate_senti_sum_in_parallel(data_to_share, afinn, grid)
    #         senti_sums_splits.append(senti_sum)
    #     senti_sums_total = reduce(gather_result, senti_sums_splits) # reduce adding up the splitted tweet counts and tweet scores
    #     senti_sums_total['#Overall Sentiment Score'] = senti_sums_total['#Overall Sentiment Score'].apply(intToPositiveStr) # change int to string with signs
    #     print('The sentimental sum of twitter dataset is:')
    #     print(senti_sums_total)
    #     i = 0
    #     for tweet in ijson.items(f, 'rows.item'):
    #         if i % size == rank:
    #             senti_sum = calculate_senti_sum_in_parallel(tweet, afinn, grid)
    #             if senti_sums_chunk.empty:
    #                 senti_sums_chunk = senti_sum
    #             else:
    #                 senti_sums_chunk = gather_result(senti_sums_chunk, senti_sum)
    #             # senti_sums_splits.append(senti_sum)
    #             # if len(senti_sums_splits) == m:
    #             #     senti_sums_chunk = reduce(gather_result, senti_sums_splits)
    #             #     senti_sums_splits = [senti_sums_chunk]
    #         i += 1
    # # senti_sums_chunk = reduce(gather_result, senti_sums_splits)
    # senti_sums_total = comm.reduce(senti_sums_chunk, op=gather_result, root=0)
    # if rank == 0:
    #     senti_sums_total['#Overall Sentiment Score'] = senti_sums_total['#Overall Sentiment Score'].apply(intToPositiveStr) # change int to string with signs
    #     print('The sentimental sum of twitter dataset is:')
    #     print(senti_sums_total)
            



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

def gather_dict(x,y):
    return {**x, **y}

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
    
    # start_t = datetime.now().timestamp()
    afinn = generate_Affin_Dict(args.afinn, comm, size, rank)
    grid = generate_grid_dict(args.grid, comm, size, rank)

    # afinn, grid = None, None
    # if rank == 0 :
    #     afinn = generate_Affin_Dict(args.afinn)
    #     grid = generate_grid_dict(args.grid)

    # afinn = comm.bcast(afinn, root=0) # broadcast afinn dict to the other members of the group
    # grid = comm.bcast(grid, root=0) # broadcast grid dict to the other members of the group
    # maxW = comm.bcast(maxW, root=0) # broadcast maxW to the other members of the group

    split_data_and_process(args.tweet, size, comm, rank, afinn, grid)
    # if rank == 0:
    #     print('Time elapsed: %s' % (datetime.now().timestamp() - start_t))

if __name__ == "__main__":
    main()
