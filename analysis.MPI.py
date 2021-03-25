import json
from mpi4py import MPI
import math
import pandas as pd
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

#########################################
# param:
#     address: the address of melbourne grid data
# return:
#     grid: the nested dict of grid information. A grid is conceptualized 
#           by its xmin, xmax, ymin, ymax properties and named by Id
########################################## 
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


#########################################
# param:
#     data: the data to be processed on each progress
#     afinn: the dict to map sentimental vocabs into num
#     grid: the dict of grid information 
# return:
#     sentiment_sums: the sum-up sentimental scores for each grid. 
#                     Eg. Cell -> #Total Tweets, #Overall Sentiment Score
########################################## 
def calculate_senti_sum_in_parallel(data, afinn, grid):
    sentiment_sums = []
    for i in range(len(grid.keys())):
        sentiment_sums.append([mapping_id_to_gc(i), 0, 0])
    special_ends = ['!', ',', '?', '.', '\'', '\"']
    for row in data:
        location = row['value']['geometry']['coordinates']
        xPos = location[0]
        yPos = location[1]
        grid_code = ''
        for key, value in grid.items():
            xMin = value['xmin']
            xMax = value['xmax']
            yMin = value['ymin']
            yMax = value['ymax']
            x_offset = 0 if xPos != xMin else -1
            y_offset = 0 if yPos != yMin else 1
            if xPos >= xMin and xPos <= xMax and yPos >= yMin and yPos < yMax:
                numCode = str(int(key[1]) + x_offset) if int(key[1]) + x_offset >= 1 else key[1]
                alphaCode = chr(ord(key[0]) + y_offset) if ord(key[0]) + y_offset <= 68 else key[0]
                grid_code = grid_code + alphaCode + numCode
        sentiment_sums[mapping_gc_to_id(grid_code)][1] += 1
        text = row['value']['properties']['text'].lower()
        words = text.split()
        for word in words:
            while word[-1] in special_ends and len(word) > 1:
                word = word[0:len(word)-1]
            if word in afinn.keys():
                sentiment_sums[mapping_gc_to_id(grid_code)][2] += afinn[word]
    sentiment_sums = pd.DataFrame(sentiment_sums, columns=['Cell', '#Total Tweets', '#Overall Sentiment Score'])
    return sentiment_sums

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

#########################################
# param:
#     path: the path of data to read
#     size: MPI.COMM_WORLD.Get_size()
# return:
#     almost evenly split json file into ${size} pieces
#     and return the array of splitted data      
########################################## 

def split_data(path, size):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        array_to_share=[[] for i in range(size)]
        rows = data['rows']
        line = 0
        for row in rows:
            array_to_share[line % size].append(row)
            line += 1
    return array_to_share

#########################################
# param:
#     x: the first sentiment sums 
#     y: the second sentiment sums
# return:
#     x: the new sentiment sums which merges the scores
#        between previous x and y     
########################################## 
def gather_result(x,y):
    x['#Total Tweets'] = x['#Total Tweets'] + y['#Total Tweets']
    x['#Overall Sentiment Score'] = x['#Overall Sentiment Score'] + y['#Overall Sentiment Score']
    return x

def read_twitter_data(address):
    tweets = []
    with open(address, 'r', encoding='utf-8') as f:
        data = json.load(f)
        tweets = data['rows']
    return tweets


if __name__ == "__main__":

    comm = MPI.COMM_WORLD  # the global communicator
    rank = comm.Get_rank() # the index of member
    size = comm.Get_size() # the total number of members

    address_1 = 'AFINN.txt'
    address_2 = 'melbGrid.json'
    address_3 = 'tinyTwitter.json'
    address_4 = 'smallTwitter.json'

    afinn, grid, data_to_share = None, None, None

    # broadcast Affin and Grid to every member in group, and then;
    # scatter Twitter data to every member in group to process (including root), and then;
    # gather data and return the result
    if rank == 0 :
        afinn = generate_Affin_Dict(address_1)
        grid = generate_grid_dict(address_2)
        data_to_share = split_data(address_4, size)  ## use smallTwitter in this demo

    afinn = comm.bcast(afinn, root=0) # broadcast afinn dict to the other members of the group
    grid = comm.bcast(grid, root=0) # broadcast grid dict to the other members of the group
    data_to_process = comm.scatter(data_to_share, root=0) # scatter the splitted data chunks to each member

    senti_sum = calculate_senti_sum_in_parallel(data_to_process, afinn, grid) # do parallel computing of sentimental scores
    senti_sums = comm.reduce(senti_sum, op=gather_result, root=0) # change comm.gather to comm.reduce

    if rank == 0:
        print('The sentimental sum of twitter dataset is:')
        print(senti_sums)
        #print('The sentimental sum of small twitter dataset is: %s' % small_sums['C2'])``