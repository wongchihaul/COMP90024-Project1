import json
import ijson

#########################################
# param:
#     address: the address of affin matchin vocabs
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
#     address: the address of data to read
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
#     address: the address of data to read
#     afinn: the dict to map sentimental vocabs into num
#     grid: the dict of grid information 
# return:
#     sentiment_sums: the sum-up sentimental scores for each grid. 
#                     Eg. Cell -> #Total Tweets, #Overall Sentiment Score
########################################## 
def calculate_senti_sum(address, afinn, grid):
    sentiment_sums = {}
    for key in grid.keys():
        sentiment_sums[key] = {'#Total Tweets' : 0, '#Overall Sentiment Score' : 0} # initialized the result nested dict
    special_ends = ['!', ',', '?', '.', '\'', '\"'] # the special ending symbols which can be tolerated
    with open(address, 'r') as f:
        data = json.load(f)
        rows = data['rows'] # each row represents a tweet
        for row in rows:
            location = row['value']['geometry']['coordinates'] # location = [xpos, ypos]
            xPos = location[0]
            yPos = location[1]
            grid_code = ''
            for key, value in grid.items():
                # obtain the location properties to define a grid
                xMin = value['xmin']
                xMax = value['xmax']
                yMin = value['ymin']
                yMax = value['ymax']
                x_offset = 0 if xPos != xMin else -1 # Tweet position on B1/B2, select B1, the left one
                y_offset = 0 if yPos != yMin else 1 # Tweet position on A1/B1, select B1, the bottom one
                if xPos >= xMin and xPos <= xMax and yPos >= yMin and yPos < yMax:
                    # E.g. 'A1' = grid_code('') + alphacode('A') + numCode('1')
                    numCode = str(int(key[1]) + x_offset) if int(key[1]) + x_offset >= 1 else key[1] 
                    alphaCode = chr(ord(key[0]) + y_offset) if ord(key[0]) + y_offset <= 68 else key[0]
                    grid_code = grid_code + alphaCode + numCode
            sentiment_sums[grid_code]['#Total Tweets'] += 1 # Adding one more tweet on the detected grid
            text = row['value']['properties']['text'].lower() # lower case requirements
            words = text.split()
            for word in words:
                while word[-1] in special_ends and len(word) > 1:
                    word = word[0:len(word)-1] # if the last character is a special symbol, get rid of the character
                if word in afinn.keys():
                    sentiment_sums[grid_code]['#Overall Sentiment Score'] += afinn[word] # Adding the sentimental score for the extracted word
    return sentiment_sums


if __name__ == "__main__":

    address_1 = 'AFINN.txt'
    address_2 = 'melbGrid.json'
    address_3 = 'tinyTwitter.json'
    address_4 = 'smallTwitter.json'

    with open(address_3, 'r', encoding='utf-8') as f:
        for prefix, event, value in ijson.parse(f):
            if prefix.endswith('.geometry.coordinates.item'):
                print(value)

    # afinn = generate_Affin_Dict(address_1)
    # grid = generate_grid_dict(address_2)
    # tiny_sum = calculate_senti_sum(address_3, afinn, grid)
    # small_sum = calculate_senti_sum(address_4, afinn, grid)

    # print('The sentimental sum of tiny twitter dataset is: %s' % tiny_sum)
    # print('The sentimental sum of small twitter dataset is: %s' % small_sum)

    # with open('tiny.json', 'w') as t :
    #     json.dump(tiny_sum, t)
    # with open('small.json', 'w') as s:
    #     json.dump(small_sum, s)
    